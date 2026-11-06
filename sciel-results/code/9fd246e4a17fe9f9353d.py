"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)

# Precompute indices for triples
TRIPLE_INDEX = {t: i for i, t in enumerate(ALL_TRIPLES)}

# Precompute the K(2,2,2) instances
# An instance is defined by 6 vertices and a partition into 3 pairs.
# We will map each instance to a list of 8 triple indices.
# And for each triple, we store the list of instance indices it belongs to.

def _precompute_instances():
    instances = []
    triple_to_instances = [[] for _ in range(N_TRIPLES)]
    
    # Iterate over all 6-vertex subsets
    for vertices in combinations(range(V), 6):
        # There are 15 partitions of 6 elements into 3 pairs
        # We can generate them by pairing the first element with any of the other 5,
        # then the first unpaired with any of the remaining 3, etc.
        # Or simply iterate over all perfect matchings.
        
        # A simpler way: 
        # A partition is determined by which vertex is paired with v0, 
        # which with the next unpaired, etc.
        # Let's just generate all 15 matchings for a set of 6.
        
        # Matchings of 6 items {0,1,2,3,4,5}
        # 1. (0,1)(2,3)(4,5)
        # 2. (0,1)(2,4)(3,5)
        # ...
        
        # We can hardcode the 15 matchings for indices 0..5
        matchings = [
            [(0,1),(2,3),(4,5)],
            [(0,1),(2,4),(3,5)],
            [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)],
            [(0,2),(1,4),(3,5)],
            [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)],
            [(0,3),(1,4),(2,5)],
            [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)],
            [(0,4),(1,3),(2,5)],
            [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)],
            [(0,5),(1,3),(2,4)],
            [(0,5),(1,4),(2,3)]
        ]
        
        for match in matchings:
            # The pairs are (vertices[match[i][0]], vertices[match[i][1]])
            # Let the pairs be P0, P1, P2
            p0 = (vertices[match[0][0]], vertices[match[0][1]])
            p1 = (vertices[match[1][0]], vertices[match[1][1]])
            p2 = (vertices[match[2][0]], vertices[match[2][1]])
            
            # The 8 triples are all combinations of one from each pair
            edges = []
            for a in p0:
                for b in p1:
                    for c in p2:
                        t = tuple(sorted((a, b, c)))
                        edges.append(TRIPLE_INDEX[t])
            
            inst_id = len(instances)
            instances.append(edges)
            
            for t_idx in edges:
                triple_to_instances[t_idx].append(inst_id)
                
    return instances, triple_to_instances

INSTANCES, TRIPLE_TO_INSTANCES = _precompute_instances()
N_INSTANCES = len(INSTANCES)

def _greedy_with_counts(deadline):
    """Greedy construction using instance counts."""
    # cnt[k] = number of edges from instance k in current solution
    cnt = [0] * N_INSTANCES
    in_solution = [False] * N_TRIPLES
    solution = []
    
    # Random order of triples
    order = list(range(N_TRIPLES))
    random.shuffle(order)
    
    for t_idx in order:
        if time.time() >= deadline:
            break
        
        if in_solution[t_idx]:
            continue
            
        # Check if adding t_idx is valid
        # It is valid if for all instances k containing t_idx, cnt[k] < 7
        # (because adding it will make it 8 if it was 7)
        valid = True
        for k in TRIPLE_TO_INSTANCES[t_idx]:
            if cnt[k] >= 7:
                valid = False
                break
        
        if valid:
            in_solution[t_idx] = True
            solution.append(ALL_TRIPLES[t_idx])
            for k in TRIPLE_TO_INSTANCES[t_idx]:
                cnt[k] += 1
                
    return solution

def _local_search(solution, deadline):
    """Try to improve the solution by swapping triples."""
    if not solution:
        return solution
    
    # Rebuild counts and in_solution array
    in_solution = [False] * N_TRIPLES
    cnt = [0] * N_INSTANCES
    sol_set = set()
    for t in solution:
        t_idx = TRIPLE_INDEX[tuple(sorted(t))]
        in_solution[t_idx] = True
        sol_set.add(t_idx)
        for k in TRIPLE_TO_INSTANCES[t_idx]:
            cnt[k] += 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to find a swap: remove t_rem, add t_add
        # We look for t_add not in solution that is blocked by some instance k.
        # If we remove t_rem which is in k, does it unblock t_add?
        
        # Optimization: Iterate over blocked triples
        blocked = []
        for t_idx in range(N_TRIPLES):
            if in_solution[t_idx]:
                continue
            for k in TRIPLE_TO_INSTANCES[t_idx]:
                if cnt[k] >= 7:
                    blocked.append((t_idx, k))
                    break
        
        random.shuffle(blocked)
        
        for t_add, k_block in blocked:
            if time.time() >= deadline:
                break
            
            # We need to remove a triple t_rem that is in instance k_block
            # and is in the solution.
            # Also, removing t_rem must not break other constraints for t_add?
            # Actually, we just need to check if t_add becomes valid after removing t_rem.
            
            # Find candidates for t_rem in instance k_block
            # Instance k_block has 8 edges.
            # We need one of them to be in the solution.
            candidates = []
            for t_idx in INSTANCES[k_block]:
                if in_solution[t_idx]:
                    candidates.append(t_idx)
            
            if not candidates:
                continue
                
            random.shuffle(candidates)
            
            for t_rem in candidates:
                # Simulate removal
                # Decrement counts for t_rem
                for k in TRIPLE_TO_INSTANCES[t_rem]:
                    cnt[k] -= 1
                    
                # Check if t_add is now valid
                valid = True
                for k in TRIPLE_TO_INSTANCES[t_add]:
                    if cnt[k] >= 7:
                        valid = False
                        break
                        
                if valid:
                    # Perform swap
                    in_solution[t_rem] = False
                    in_solution[t_add] = True
                    solution.remove(ALL_TRIPLES[t_rem])
                    solution.append(ALL_TRIPLES[t_add])
                    
                    # Update counts for t_add
                    for k in TRIPLE_TO_INSTANCES[t_add]:
                        cnt[k] += 1
                        
                    improved = True
                    break
                else:
                    # Revert counts for t_rem
                    for k in TRIPLE_TO_INSTANCES[t_rem]:
                        cnt[k] += 1
                    continue
                    
            if improved:
                break
                
    return solution

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_solution = []
    best_len = 0
    
    # Run multiple greedy + local search attempts
    for i in range(100): # Max attempts, will stop early if time runs out
        if time.time() >= deadline - 10: # Reserve 10s for final validation
            break
            
        sol = _greedy_with_counts(deadline - 10)
        
        # Local search
        sol = _local_search(sol, deadline - 10)
        
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol
            # Optional: break if we hit a high score, but let's keep trying
            # if best_len > 500: break # Threshold unknown, keep going
            
    # Final Validation
    if not best_solution:
         # Fallback: Star
         return [list(t) for t in ALL_TRIPLES if 0 in t]
         
    best_set = set(tuple(sorted(t)) for t in best_solution)
    valid = True
    for k in range(N_INSTANCES):
        edges = INSTANCES[k]
        # Check if all 8 are present
        if all(TRIPLE_INDEX[t] in best_set for t in edges):
             # Wait, best_set contains tuples, TRIPLE_INDEX[t] is int.
             # Let's just check tuples.
             pass
             
    # Correct validation loop
    best_set_tuples = set(tuple(sorted(t)) for t in best_solution)
    for k in range(N_INSTANCES):
        edges = INSTANCES[k]
        # edges are indices
        all_present = True
        for t_idx in edges:
             if ALL_TRIPLES[t_idx] not in best_set_tuples:
                 all_present = False
                 break
        if all_present:
             valid = False
             break
             
    if not valid:
         # Fallback
         return [list(t) for t in ALL_TRIPLES if 0 in t]
         
    return best_solution
