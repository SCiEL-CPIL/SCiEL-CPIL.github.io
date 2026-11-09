import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    def exact_independence_number(triples):
        """
        Computes the exact independence number of the hypergraph defined by triples.
        This is equivalent to the Maximum Independent Set (MIS) of the shadow graph,
        where two points are connected if they share a triple.
        Uses a branch-and-bound solver with bitset representation and pruning.
        """
        if not triples:
            return 16
            
        # 1. Build the shadow graph adjacency bitmasks
        # adj[i] is a bitmask of points j such that i and j are in some triple.
        adj = [0] * 16
        for a, b, c in triples:
            # a is connected to b and c
            adj[a] |= (1 << b)
            adj[a] |= (1 << c)
            # b is connected to a and c
            adj[b] |= (1 << a)
            adj[b] |= (1 << c)
            # c is connected to a and b
            adj[c] |= (1 << a)
            adj[c] |= (1 << b)
            
        # Note: The "independent set" in the hypergraph sense is a set of points
        # such that NO triple is fully contained.
        # Wait, the task says: "score is the largest point set containing no triple".
        # This means we are looking for a set S such that for all T in triples, T is not a subset of S.
        # This is NOT the independent set of the shadow graph (which forbids pairs).
        # This is the independence number of the 3-uniform hypergraph.
        
        # Let's re-evaluate the problem definition.
        # "largest point set containing no triple"
        # If S contains {a, b, c} where {a, b, c} is a triple, then S is NOT valid.
        # So we want max |S| such that for all T in triples, T not subset S.
        
        # This is different from MIS on the shadow graph.
        # MIS on shadow graph: max |S| such that no edge (pair) is in S.
        # Here: max |S| such that no hyperedge (triple) is in S.
        
        # So the "forbidden" structures are triples.
        # We can use a similar branch and bound, but the conflict check is "does this set contain any triple?".
        
        # Precompute triple masks for fast conflict checking
        triple_masks = []
        for a, b, c in triples:
            mask = (1 << a) | (1 << b) | (1 << c)
            triple_masks.append(mask)
            
        # Sort triple masks to help with pruning? Not strictly necessary for bitset AND.
        
        best_size = 0
        
        # Branch and bound
        # State: current_mask (bitmask of selected points), size, start_idx
        # Pruning:
        # 1. If size + (16 - start_idx) <= best_size: prune
        # 2. If current_mask contains any triple: prune (should not happen if we check before adding)
        
        # To speed up "does current_mask contain any triple?", we can check:
        # for tm in triple_masks: if (current_mask & tm) == tm: return False
        
        # Optimization:
        # Instead of checking all triples at every step, we can order the search
        # and prune if adding a point completes a triple.
        
        # Let's use a recursive search with bitmasks.
        
        def backtrack(mask, size, start_idx):
            nonlocal best_size
            
            if size > best_size:
                best_size = size
                
            # Pruning 1: Upper bound
            if size + (16 - start_idx) <= best_size:
                return
                
            # Try adding points from start_idx to 15
            for i in range(start_idx, 16):
                if not (mask & (1 << i)): # i is not already in mask (should be true by construction of start_idx)
                    new_mask = mask | (1 << i)
                    
                    # Check if new_mask is independent (contains no triple)
                    # We can optimize this: only check triples that include i and are subsets of new_mask.
                    # But checking all 16*15*14/6 = 560 triples is fast if we use bitsets.
                    # However, we can be smarter.
                    
                    is_valid = True
                    for tm in triple_masks:
                        if (new_mask & tm) == tm:
                            is_valid = False
                            break
                            
                    if is_valid:
                        backtrack(new_mask, size + 1, i + 1)
                    else:
                        # If adding i makes it invalid, we cannot add i.
                        # Can we prune the rest?
                        # Not necessarily, because other points might be valid.
                        pass
                        
        backtrack(0, 0, 0)
        return best_size

    def is_linear(triples):
        """Check if the set of triples is linear (no pair repeated)."""
        used_pairs = set()
        for a, b, c in triples:
            pairs = [(a, b), (b, c), (a, c)]
            for p in pairs:
                p_sorted = tuple(sorted(p))
                if p_sorted in used_pairs:
                    return False
                used_pairs.add(p_sorted)
        return True

    def greedy_packing(seed):
        random.seed(seed)
        points = list(range(16))
        all_triples = list(combinations(points, 3))
        random.shuffle(all_triples)
        
        triples = []
        used_pairs = set()
        
        for t in all_triples:
            a, b, c = t
            p1 = (a, b) if a < b else (b, a)
            p2 = (b, c) if b < c else (c, b)
            p3 = (a, c) if a < c else (c, a)
            
            if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
                triples.append([a, b, c])
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)
                
        return triples

    def local_search(system, max_steps, time_limit):
        """
        Improve the system by swapping points in triples to reduce independence number.
        Uses exact evaluation.
        """
        current_system = [list(t) for t in system]
        current_score = exact_independence_number(current_system)
        
        # Precompute pair usage for quick linearity check
        def get_used_pairs(sys):
            used = set()
            for a, b, c in sys:
                used.add((a, b) if a < b else (b, a))
                used.add((b, c) if b < c else (c, b))
                used.add((a, c) if a < c else (c, a))
            return used

        used_pairs = get_used_pairs(current_system)
        
        step = 0
        while step < max_steps and time.time() < time_limit:
            step += 1
            if time.time() > time_limit:
                break
                
            improved = False
            # Randomly select a triple to modify
            idx = random.randint(0, len(current_system) - 1)
            t = current_system[idx]
            
            # Try replacing each point in t with a random other point
            for k in range(3):
                old_point = t[k]
                # Candidates: points not in t
                candidates = [p for p in range(16) if p not in t]
                random.shuffle(candidates)
                
                for new_point in candidates:
                    # Construct new triple
                    new_t = t.copy()
                    new_t[k] = new_point
                    # Ensure no duplicate points in triple (should be guaranteed by candidates)
                    if len(set(new_t)) < 3:
                        continue
                        
                    # Check linearity
                    a, b, c = t
                    old_pairs = [
                        (a, b) if a < b else (b, a),
                        (b, c) if b < c else (c, b),
                        (a, c) if a < c else (c, a)
                    ]
                    
                    a, b, c = new_t
                    new_pairs = [
                        (a, b) if a < b else (b, a),
                        (b, c) if b < c else (c, b),
                        (a, c) if a < c else (c, a)
                    ]
                    
                    # Check if any new pair is already in used_pairs (excluding the ones we just removed)
                    # Temporarily remove old pairs
                    for p in old_pairs:
                        used_pairs.remove(p)
                        
                    valid = True
                    for p in new_pairs:
                        if p in used_pairs:
                            valid = False
                            break
                            
                    if valid:
                        # Add new pairs
                        for p in new_pairs:
                            used_pairs.add(p)
                        
                        # Update system
                        current_system[idx] = new_t
                        
                        # Evaluate
                        new_score = exact_independence_number(current_system)
                        
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            break
                        else:
                            # Revert
                            for p in new_pairs:
                                used_pairs.remove(p)
                            for p in old_pairs:
                                used_pairs.add(p)
                            current_system[idx] = t
                    else:
                        # Revert removal of old pairs
                        for p in old_pairs:
                            used_pairs.add(p)
                            
                    if improved:
                        break
                if improved:
                    break
            if not improved and step % 50 == 0:
                # If stuck for a while, maybe break to save time for other runs?
                # Or just continue.
                pass
                    
        return current_system, current_score

    best_system = None
    best_score = 16
    
    # Run multiple instances of: Greedy + Local Search
    num_runs = 200 # Increase runs since evaluation is faster/more reliable
    for i in range(num_runs):
        if time.time() > deadline - 10.0:
            break
            
        # Generate initial packing
        init_system = greedy_packing(i)
        
        # Local search
        # Time limit per local search: dynamic based on remaining time
        remaining = deadline - time.time()
        if remaining < 1.0: break
        
        # Allocate a small fraction of remaining time to each search
        # But ensure we can do at least a few.
        search_limit = min(0.5, remaining / 10.0)
        
        search_start = time.time()
        optimized_system, score = local_search(init_system, max_steps=5000, time_limit=search_start + search_limit)
        
        if score < best_score:
            best_score = score
            best_system = optimized_system
            
            if best_score <= 5:
                break

    if best_system is None:
        # Fallback: a valid linear system (e.g. disjoint blocks)
        # 5 disjoint blocks of 3 + 1 leftover point?
        # 16 points. 5 blocks of 3 = 15 points.
        # Independence number would be 16 - 0 = 16? No.
        # If blocks are disjoint, a set can pick 2 from each block.
        # 5 blocks * 2 = 10. Plus the leftover point = 11.
        # So score 11 is achievable easily.
        # Let's try to get lower.
        
        # A better fallback:
        # Construct a specific known good structure if possible, or just return the greedy result.
        # The greedy result is always valid.
        init_system = greedy_packing(0)
        return init_system
        
    return best_system
# EVOLVE-BLOCK-END
