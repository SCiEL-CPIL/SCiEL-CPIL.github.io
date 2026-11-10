"""Sparse squares via collision-graph branch-and-bound and factor decomposition."""
import time
import random
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    MAX_DEG = 40
    NUM_TERMS = 12
    MAX_VAL = 64

    def evaluate(positions, values):
        """Count nonzero terms in P^2."""
        dense = np.zeros(MAX_DEG + 1, dtype=np.int64)
        for p, v in zip(positions, values):
            if 0 <= p <= MAX_DEG:
                dense[p] = v
            else:
                return 1000
        sq = np.convolve(dense, dense, mode='full')
        return int(np.count_nonzero(sq))

    def fast_eval(positions, values):
        """Faster evaluation using sparse convolution."""
        # Build sparse pairs
        pairs = [(p, v) for p, v in zip(positions, values) if v != 0]
        if len(pairs) != NUM_TERMS:
            return 1000
        # Compute all pairwise products
        coeff_dict = {}
        for i in range(len(pairs)):
            pi, vi = pairs[i]
            for j in range(i, len(pairs)):
                pj, vj = pairs[j]
                deg = pi + pj
                prod = vi * vj
                if i == j:
                    coeff_dict[deg] = coeff_dict.get(deg, 0) + prod
                else:
                    coeff_dict[deg] = coeff_dict.get(deg, 0) + 2 * prod
        return len(coeff_dict)

    def collision_graph(positions):
        """
        Build collision graph: vertices are positions, edges indicate 
        degree collisions in P^2. Returns adjacency as dict of sets.
        """
        # For each pair (i,j) with i<=j, the product contributes to degree pos[i]+pos[j]
        # Two products collide if they map to the same degree
        # We want to find which position pairs create unique degrees
        degree_map = {}  # degree -> list of (i,j) pairs
        for i in range(NUM_TERMS):
            for j in range(i, NUM_TERMS):
                deg = positions[i] + positions[j]
                if deg not in degree_map:
                    degree_map[deg] = []
                degree_map[deg].append((i, j))
        
        # Adjacency: two indices are "connected" if they participate in a collision
        adj = {i: set() for i in range(NUM_TERMS)}
        for deg, pairs in degree_map.items():
            if len(pairs) > 1:
                # All indices involved in this collision are connected
                involved = set()
                for (i, j) in pairs:
                    involved.add(i)
                    involved.add(j)
                for i in involved:
                    for j in involved:
                        if i != j:
                            adj[i].add(j)
        return adj, degree_map

    def branch_and_bound_support(positions, time_limit):
        """
        Branch-and-bound to find value assignments that maximize cancellations.
        Uses the collision graph to guide search.
        """
        adj, degree_map = collision_graph(positions)
        
        # Sort degrees by collision count (most constrained first)
        constrained_degrees = sorted(
            [d for d, pairs in degree_map.items() if len(pairs) > 1],
            key=lambda d: -len(degree_map[d])
        )
        
        best_score = fast_eval(positions, [1]*NUM_TERMS)
        best_values = [1]*NUM_TERMS
        
        # Try small integer values first
        value_candidates = [1, -1, 2, -2, 3, -3]
        
        # Recursive search with pruning
        def search(idx, current_values, current_score):
            nonlocal best_score, best_values
            
            if time.time() > time_limit:
                return
            
            if idx == NUM_TERMS:
                if current_score < best_score:
                    best_score = current_score
                    best_values = current_values[:]
                return
            
            # Try candidate values
            for v in value_candidates:
                current_values[idx] = v
                new_score = fast_eval(positions, current_values)
                if new_score < best_score or new_score <= current_score:
                    search(idx + 1, current_values, new_score)
            
            current_values[idx] = 0
        
        search(0, [0]*NUM_TERMS, 1000)
        return best_values, best_score

    def factor_decompose_search(time_limit):
        """
        Search for polynomials P = A * B where A and B are sparser.
        This can inherit cancellation patterns from factor squares.
        """
        best_score = 1000
        best_pos = None
        best_val = None
        
        # Try decompositions: 2x6, 3x4, 4x3, 6x2
        factor_sizes = [(2, 6), (3, 4), (4, 3), (6, 2)]
        
        for size_a, size_b in factor_sizes:
            if time.time() > time_limit:
                break
            
            # Generate random sparse factors
            for trial in range(50):
                if time.time() > time_limit:
                    break
                
                # Factor A: size_a terms
                pos_a = random.sample(range(MAX_DEG + 1), size_a)
                pos_a.sort()
                val_a = [random.choice([1, -1, 2, -2]) for _ in range(size_a)]
                
                # Factor B: size_b terms
                pos_b = random.sample(range(MAX_DEG + 1), size_b)
                pos_b.sort()
                val_b = [random.choice([1, -1, 2, -2]) for _ in range(size_b)]
                
                # Compute P = A * B
                dense_a = np.zeros(MAX_DEG + 1, dtype=np.int64)
                for p, v in zip(pos_a, val_a):
                    dense_a[p] = v
                
                dense_b = np.zeros(MAX_DEG + 1, dtype=np.int64)
                for p, v in zip(pos_b, val_b):
                    dense_b[p] = v
                
                dense_p = np.convolve(dense_a, dense_b, mode='full')
                
                # Check if P has exactly NUM_TERMS nonzero terms
                nonzero_indices = np.nonzero(dense_p)[0]
                if len(nonzero_indices) != NUM_TERMS:
                    continue
                if len(nonzero_indices) > MAX_DEG:
                    continue
                
                positions = nonzero_indices.tolist()
                values = dense_p[nonzero_indices].tolist()
                
                # Check values are in range
                if any(abs(v) > MAX_VAL for v in values):
                    continue
                
                score = evaluate(positions, values)
                if score < best_score:
                    best_score = score
                    best_pos = positions
                    best_val = values
        
        return best_pos, best_val, best_score

    def local_search(positions, values, time_limit):
        """Local search refinement."""
        current_pos = positions[:]
        current_val = values[:]
        current_score = fast_eval(current_pos, current_val)
        
        best_pos = current_pos[:]
        best_val = current_val[:]
        best_score = current_score
        
        start_t = time.time()
        
        while time.time() - start_t < time_limit:
            # Try moving one position
            improved = False
            for idx in range(NUM_TERMS):
                for new_p in range(MAX_DEG + 1):
                    if new_p in current_pos and new_p != current_pos[idx]:
                        continue
                    if new_p == current_pos[idx]:
                        continue
                    
                    old_p = current_pos[idx]
                    current_pos[idx] = new_p
                    
                    new_score = fast_eval(current_pos, current_val)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_pos = current_pos[:]
                            best_val = current_val[:]
                    else:
                        current_pos[idx] = old_p
                
                if time.time() - start_t > time_limit:
                    break
            
            if not improved:
                # Try changing values
                for idx in range(NUM_TERMS):
                    for new_v in [-3, -2, -1, 1, 2, 3, 4, -4]:
                        if new_v == current_val[idx]:
                            continue
                        old_v = current_val[idx]
                        current_val[idx] = new_v
                        new_score = fast_eval(current_pos, current_val)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_pos = current_pos[:]
                                best_val = current_val[:]
                        else:
                            current_val[idx] = old_v
                    
                    if time.time() - start_t > time_limit:
                        break
            
            if not improved:
                # No improvement, try random restart
                if random.random() < 0.3:
                    idx = random.randint(0, NUM_TERMS - 1)
                    current_pos[idx] = random.randint(0, MAX_DEG)
                    if len(set(current_pos)) != NUM_TERMS:
                        current_pos[idx] = random.randint(0, MAX_DEG)
                    current_val[idx] = random.choice([-2, -1, 1, 2])
                    current_score = fast_eval(current_pos, current_val)
                else:
                    break
        
        return best_pos, best_val, best_score

    # Main algorithm
    global_best_score = 1000
    global_best_pos = None
    global_best_val = None
    
    # Phase 1: Factor decomposition search
    remaining = deadline - time.time() - 5
    time_for_factors = remaining * 0.3
    f_pos, f_val, f_score = factor_decompose_search(time_for_factors)
    if f_pos is not None and f_score < global_best_score:
        global_best_score = f_score
        global_best_pos = f_pos
        global_best_val = f_val
    
    # Phase 2: Branch-and-bound on good supports
    remaining = deadline - time.time() - 5
    time_for_bnb = remaining * 0.3
    
    # Generate candidate supports
    candidate_supports = []
    
    # From factor search
    if f_pos is not None:
        candidate_supports.append(f_pos)
    
    # Structured supports
    for step in [1, 2, 3, 4]:
        if (NUM_TERMS - 1) * step <= MAX_DEG:
            pos = list(range(0, NUM_TERMS * step, step))
            candidate_supports.append(pos)
    
    # Random supports
    for _ in range(20):
        candidate_supports.append(random.sample(range(MAX_DEG + 1), NUM_TERMS))
    
    # Try B&B on each candidate
    time_per_bnb = time_for_bnb / max(len(candidate_supports), 1)
    for support in candidate_supports:
        if time.time() > deadline - 5:
            break
        vals, score = branch_and_bound_support(support, time_per_bnb)
        if score < global_best_score:
            global_best_score = score
            global_best_pos = support
            global_best_val = vals
    
    # Phase 3: Local search refinement
    remaining = deadline - time.time() - 2
    if remaining > 1 and global_best_pos is not None:
        l_pos, l_val, l_score = local_search(global_best_pos, global_best_val, remaining)
        if l_score < global_best_score:
            global_best_score = l_score
            global_best_pos = l_pos
            global_best_val = l_val
    
    # Fallback
    if global_best_pos is None:
        global_best_pos = random.sample(range(MAX_DEG + 1), NUM_TERMS)
        global_best_val = [1]*NUM_TERMS
        global_best_score = fast_eval(global_best_pos, global_best_val)
    
    # Construct final coefficient list
    max_pos = max(global_best_pos)
    coeff_list = [0] * (max_pos + 1)
    for p, v in zip(global_best_pos, global_best_val):
        coeff_list[p] = v
    
    if len(coeff_list) > 41:
        coeff_list = coeff_list[:41]
    
    return coeff_list

# EVOLVE-BLOCK-END
