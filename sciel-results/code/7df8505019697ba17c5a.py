import time
import random
import itertools
import math

# Precompute all triples and compatibility
N = 16
ALL_TRIPLES = []
TRIPLE_MASKS = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            mask = (1 << a) | (1 << b) | (1 << c)
            ALL_TRIPLES.append((a, b, c))
            TRIPLE_MASKS.append(mask)

NUM_TRIPLES = len(ALL_TRIPLES)

# Precompute compatibility: compat[i] is a set of indices j such that triple i and j are compatible
COMPAT = [set() for _ in range(NUM_TRIPLES)]
for i in range(NUM_TRIPLES):
    for j in range(i + 1, NUM_TRIPLES):
        # Compatible if intersection size <= 1
        # Intersection size is popcount(mask_i & mask_j)
        inter = TRIPLE_MASKS[i] & TRIPLE_MASKS[j]
        if bin(inter).count('1') <= 1:
            COMPAT[i].add(j)
            COMPAT[j].add(i)

# Precompute for faster alpha calculation
# For each subset of vertices (as bitmask), store whether it's independent given a set of edges
# We'll use a different approach: for each edge, precompute which vertex subsets contain it

# For faster checking: given a set of edge masks, check if subset S is independent
# S is independent iff no edge mask is a subset of S
# i.e., for all e in edges, (e & S) != e

# Precompute popcount for all 16-bit masks
POPCOUNT = [0] * (1 << N)
for i in range(1 << N):
    POPCOUNT[i] = bin(i).count('1')

def calc_alpha_fast(edge_masks):
    """
    Calculate the independence number using bitmask operations.
    edge_masks is a list of integer masks.
    """
    if not edge_masks:
        return N
    
    # Check from largest possible size down
    for k in range(N, 0, -1):
        # Generate all subsets of size k using itertools
        for subset in itertools.combinations(range(N), k):
            mask_s = 0
            for v in subset:
                mask_s |= (1 << v)
            
            is_indep = True
            for em in edge_masks:
                if (mask_s & em) == em:
                    is_indep = False
                    break
            
            if is_indep:
                return k
            
    return 0

def calc_alpha(edge_indices):
    """
    Calculate the independence number of the hypergraph defined by edge_indices.
    Returns the size of the largest independent set.
    """
    if not edge_indices:
        return N
    
    edge_masks = [TRIPLE_MASKS[i] for i in edge_indices]
    return calc_alpha_fast(edge_masks)

def greedy_construct():
    """
    Construct a maximal linear system using a greedy randomized approach.
    Returns list of edge indices.
    """
    order = list(range(NUM_TRIPLES))
    random.shuffle(order)
    
    selected = []
    selected_set = set()
    
    for idx in order:
        # Check if idx is compatible with all selected
        compatible = True
        for s_idx in selected:
            if s_idx not in COMPAT[idx]:
                compatible = False
                break
        
        if compatible:
            selected.append(idx)
            selected_set.add(idx)
            
    return selected

def greedy_construct_weighted():
    """
    Greedy construction that prioritizes triples covering uncovered pairs.
    This aims to maximize coverage and minimize independence number.
    """
    # Track which pairs are covered
    # A pair (a,b) is covered if there's a triple containing both a and b
    # Actually, for linear hypergraphs, we just need to ensure no two triples share 2+ vertices
    
    # Better approach: prioritize triples that cover vertices with high degree potential
    # Or: prioritize triples that are "central" in the compatibility graph
    
    # Simple heuristic: start with random order but weight by how many compatible triples remain
    # This is complex; let's try a different approach
    
    # Approach: iterative deepening - start with a random seed triple, then greedily add
    # compatible triples, preferring those that cover more "rare" pairs
    
    # For now, use multiple restarts with different random seeds and keep the best
    order = list(range(NUM_TRIPLES))
    random.shuffle(order)
    
    selected = []
    selected_set = set()
    
    for idx in order:
        compatible = True
        for s_idx in selected:
            if s_idx not in COMPAT[idx]:
                compatible = False
                break
        
        if compatible:
            selected.append(idx)
            selected_set.add(idx)
            
    return selected

def local_search(edge_indices, max_iters=1000):
    """
    Try to improve the independence number by swapping edges.
    Supports 1-swap and 2-swap moves.
    """
    current_edges = list(edge_indices)
    current_alpha = calc_alpha(current_edges)
    
    for _ in range(max_iters):
        if len(current_edges) < 1:
            break
            
        made_move = False
        
        # Try 1-swap: remove one edge, add one compatible edge
        if len(current_edges) >= 1:
            rem_idx = random.choice(range(len(current_edges)))
            rem_edge = current_edges[rem_idx]
            
            temp_edges = current_edges[:rem_idx] + current_edges[rem_idx+1:]
            temp_set = set(temp_edges)
            
            # Find compatible candidates
            # Precompute which triples are compatible with all in temp_set
            # For speed, just try random ones
            for _ in range(20):
                cand_idx = random.randint(0, NUM_TRIPLES - 1)
                if cand_idx in temp_set:
                    continue
                
                ok = True
                for s_idx in temp_edges:
                    if s_idx not in COMPAT[cand_idx]:
                        ok = False
                        break
                
                if ok:
                    new_edges = temp_edges + [cand_idx]
                    new_alpha = calc_alpha(new_edges)
                    
                    if new_alpha < current_alpha:
                        current_edges = new_edges
                        current_alpha = new_alpha
                        made_move = True
                        break
                    elif new_alpha == current_alpha and random.random() < 0.15:
                        current_edges = new_edges
                        made_move = True
                        break
        
        # Try 2-swap: remove two edges, add two compatible edges
        if not made_move and len(current_edges) >= 2:
            rem_indices = random.sample(range(len(current_edges)), 2)
            rem_indices.sort(reverse=True)  # Sort to remove from end first
            rem_edge1 = current_edges[rem_indices[0]]
            rem_edge2 = current_edges[rem_indices[1]]
            
            temp_edges = []
            for i in range(len(current_edges)):
                if i not in rem_indices:
                    temp_edges.append(current_edges[i])
            temp_set = set(temp_edges)
            
            # Find two compatible edges to add
            for _ in range(15):
                cand1 = random.randint(0, NUM_TRIPLES - 1)
                if cand1 in temp_set:
                    continue
                
                ok1 = True
                for s_idx in temp_edges:
                    if s_idx not in COMPAT[cand1]:
                        ok1 = False
                        break
                
                if not ok1:
                    continue
                
                cand2 = random.randint(0, NUM_TRIPLES - 1)
                if cand2 in temp_set or cand2 == cand1:
                    continue
                
                ok2 = True
                for s_idx in temp_edges:
                    if s_idx not in COMPAT[cand2]:
                        ok2 = False
                        break
                
                # Also check cand1 and cand2 are compatible
                if cand1 not in COMPAT[cand2]:
                    continue
                
                if ok2:
                    new_edges = temp_edges + [cand1, cand2]
                    new_alpha = calc_alpha(new_edges)
                    
                    if new_alpha < current_alpha:
                        current_edges = new_edges
                        current_alpha = new_alpha
                        made_move = True
                        break
                    elif new_alpha == current_alpha and random.random() < 0.1:
                        current_edges = new_edges
                        made_move = True
                        break
        
        if not made_move:
            # Stuck; try a few more iterations before giving up
            continue
            
    return current_edges, current_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    best_alpha = N
    best_system = []
    
    # Phase 1: Multiple greedy + local search restarts
    num_restarts = 0
    while time.time() < deadline - 20:  # Leave 20s for final polish
        # Generate a new greedy system
        sys = greedy_construct()
        
        # Local search to improve
        sys, alpha = local_search(sys, max_iters=300)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = sys
            num_restarts += 1
            
        # If we hit a very low alpha, we might break early
        if best_alpha <= 4:
            break
            
        # Limit restarts to avoid getting stuck in bad local optima
        if num_restarts > 20 and time.time() > start_time + 60:
            break
    
    # Phase 2: Intensive local search on the best solution
    if best_system and time.time() < deadline - 5:
        sys, alpha = local_search(best_system, max_iters=2000)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = sys
    
    # Phase 3: Try to add more edges if possible (maximal check)
    if best_system and time.time() < deadline - 2:
        current_set = set(best_system)
        # Try to add any compatible edge
        for idx in range(NUM_TRIPLES):
            if idx in current_set:
                continue
            ok = True
            for s_idx in best_system:
                if s_idx not in COMPAT[idx]:
                    ok = False
                    break
            if ok:
                new_sys = best_system + [idx]
                new_alpha = calc_alpha(new_sys)
                if new_alpha <= best_alpha:
                    best_system = new_sys
                    best_alpha = new_alpha
                    current_set.add(idx)
    
    # Convert indices back to triples
    return [ALL_TRIPLES[i] for i in best_system]
