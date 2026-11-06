import time
import random
import sys
from itertools import combinations

# Increase recursion limit
sys.setrecursionlimit(100000)

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute triple indices
_triple_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

# --- Precomputation: K(2,2,2) Configurations ---
# A K(2,2,2) is defined by 3 disjoint pairs of vertices.
# Total number: C(16,6) * 15 = 8008 * 15 = 120120

def _precompute_k222s():
    triple_to_k222s = [[] for _ in range(N_TRIPLES)]
    k222_triples = []
    
    def get_matchings(verts):
        matchings = []
        v0 = verts[0]
        for i in range(1, 6):
            p1 = (v0, verts[i])
            remaining = [v for j, v in enumerate(verts) if j != 0 and j != i]
            r0, r1, r2, r3 = remaining
            matchings.append((p1, (r0, r1), (r2, r3)))
            matchings.append((p1, (r0, r2), (r1, r3)))
            matchings.append((p1, (r0, r3), (r1, r2)))
        return matchings

    for verts in combinations(range(V), 6):
        for matching in get_matchings(verts):
            p1, p2, p3 = matching
            t_indices = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        t_indices.append(_triple_idx[t])
            
            k_idx = len(k222_triples)
            k222_triples.append(t_indices)
            
            for t_idx in t_indices:
                triple_to_k222s[t_idx].append(k_idx)
                
    return triple_to_k222s, k222_triples

# Run precomputation once at module load
triple_to_k222s, k222_triples = _precompute_k222s()
N_K222 = len(k222_triples)

# Build conflict graph: two triples conflict if they are in the same K222 and together with 6 other triples complete it
# Actually, the constraint is: no K222 can have all 8 of its transversals present.
# So we need to find the maximum set of triples such that for every K222, at most 7 of its 8 triples are present.
# This is NOT a simple independent set problem! It's a hypergraph independent set (or packing problem).
# 
# Let me reconsider. The constraint is: for each K222 (which has 8 triples), we can have at most 7.
# This is a 0-1 integer program with constraints: for each K222 k, sum_{t in K222_k} x_t <= 7.
#
# This is NOT a standard maximum independent set. Let me use a different approach.
# 
# Actually, I can model this as a maximum weight independent set in a conflict graph where:
# - Vertices are triples
# - Two triples are adjacent (conflict) if they cannot both be in the solution
# But the constraint is not pairwise - it's that no 8 specific triples can all be present.
#
# Let me think differently. I'll use simulated annealing / local search with a better approach.
# Or I can use the fact that the constraint is "at most 7 out of 8" for each K222.
#
# For local search, I can:
# 1. Start with a greedy solution
# 2. Try to add triples that don't violate constraints
# 3. Try swaps (remove one, add one) to improve
# 4. Use bitsets for fast constraint checking

# Build bitset representation for fast conflict checking
# For each triple, store which K222s it belongs to
# For each K222, store the 8 triples that belong to it

# Precompute: for each triple, a bitmask of K222s it belongs to
# Since N_K222 = 120120, we need 120120 bits = 15016 bytes per triple
# 560 * 15016 = 8.4 MB, which is feasible

# Actually, let's use a simpler approach with counts and efficient local search

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    def time_left():
        return deadline - time.time()
    
    # Greedy initialization
    def greedy_init(seed):
        random.seed(seed)
        active = [False] * N_TRIPLES
        counts = [0] * N_K222
        current_size = 0
        
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        for t_idx in order:
            valid = True
            for k_idx in triple_to_k222s[t_idx]:
                if counts[k_idx] == 7:
                    valid = False
                    break
            
            if valid:
                active[t_idx] = True
                current_size += 1
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] += 1
        
        return active, counts, current_size
    
    # Local search with add and swap moves
    def local_search(active, counts, current_size, max_iterations=500000):
        improved = True
        iteration = 0
        # Track which inactive triples are blocked and why
        blocked_reasons = {}  # t_idx -> list of k_idx that are full
        
        while improved and time_left() > 2.0 and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Phase 1: Try to add any possible triple (only check previously blocked ones first)
            # First, try previously blocked triples that might now be addable
            for t_idx in list(blocked_reasons.keys()):
                if time_left() < 0.5:
                    return active, counts, current_size
                if active[t_idx]:
                    continue
                # Check if still blocked
                still_blocked = []
                for k_idx in triple_to_k222s[t_idx]:
                    if counts[k_idx] == 7:
                        still_blocked.append(k_idx)
                if not still_blocked:
                    active[t_idx] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[t_idx]:
                        counts[k_idx] += 1
                    improved = True
                    del blocked_reasons[t_idx]
                else:
                    blocked_reasons[t_idx] = still_blocked
            
            # Then check all other inactive triples
            if not improved:
                for i in range(N_TRIPLES):
                    if time_left() < 0.5:
                        return active, counts, current_size
                    if not active[i]:
                        addable = True
                        blockers = []
                        for k_idx in triple_to_k222s[i]:
                            if counts[k_idx] == 7:
                                addable = False
                                blockers.append(k_idx)
                        if addable:
                            active[i] = True
                            current_size += 1
                            for k_idx in triple_to_k222s[i]:
                                counts[k_idx] += 1
                            improved = True
                        else:
                            blocked_reasons[i] = blockers
            
            if improved:
                continue
            
            # Phase 2: Try swaps (1-for-1) - try ALL inactive, not just 200
            inactive_indices = [i for i in range(N_TRIPLES) if not active[i]]
            random.shuffle(inactive_indices)
            
            found_swap = False
            for t_add in inactive_indices:
                if time_left() < 0.5:
                    return active, counts, current_size
                
                # Use cached blockers if available
                if t_add in blocked_reasons:
                    blockers = blocked_reasons[t_add]
                else:
                    blockers = []
                    for k_idx in triple_to_k222s[t_add]:
                        if counts[k_idx] == 7:
                            blockers.append(k_idx)
                
                if not blockers:
                    active[t_add] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[t_add]:
                        counts[k_idx] += 1
                    improved = True
                    found_swap = True
                    break
                
                # Find t_remove that contains all blockers
                if len(blockers) == 1:
                    candidates = k222_triples[blockers[0]]
                else:
                    # Use set intersection for multiple blockers
                    candidates = set(k222_triples[blockers[0]])
                    for k_block in blockers[1:]:
                        candidates.intersection_update(k222_triples[k_block])
                    
                    if not candidates:
                        continue
                    candidates = list(candidates)
                
                for t_remove in candidates:
                    if active[t_remove]:
                        # Perform swap
                        active[t_remove] = False
                        for k_idx in triple_to_k222s[t_remove]:
                            counts[k_idx] -= 1
                        
                        active[t_add] = True
                        for k_idx in triple_to_k222s[t_add]:
                            counts[k_idx] += 1
                        
                        # Update blocked_reasons: t_remove is now inactive
                        blocked_reasons.pop(t_add, None)
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            if not found_swap:
                # Try 2-for-1 moves: remove 2 active, add 1 inactive
                # This can help escape local optima
                active_indices = [i for i in range(N_TRIPLES) if active[i]]
                random.shuffle(inactive_indices)
                random.shuffle(active_indices)
                
                found_2for1 = False
                for t_add in inactive_indices[:100]:
                    if time_left() < 0.5:
                        return active, counts, current_size
                    
                    if t_add in blocked_reasons:
                        blockers = blocked_reasons[t_add]
                    else:
                        blockers = []
                        for k_idx in triple_to_k222s[t_add]:
                            if counts[k_idx] == 7:
                                blockers.append(k_idx)
                    
                    if not blockers:
                        # Can add directly
                        active[t_add] = True
                        current_size += 1
                        for k_idx in triple_to_k222s[t_add]:
                            counts[k_idx] += 1
                        improved = True
                        found_2for1 = True
                        break
                    
                    # Need to remove at least one triple from each blocker
                    # Try removing pairs of active triples
                    for r1 in active_indices[:50]:
                        if time_left() < 0.5:
                            return active, counts, current_size
                        
                        # Check if removing r1 helps
                        remaining_blockers = []
                        for k_idx in blockers:
                            if counts[k_idx] > 7:
                                remaining_blockers.append(k_idx)
                            elif r1 in k222_triples[k_idx]:
                                # Would be fixed by removing r1
                                continue
                            else:
                                remaining_blockers.append(k_idx)
                        
                        if not remaining_blockers:
                            # Just remove r1 and add t_add
                            active[r1] = False
                            for k_idx in triple_to_k222s[r1]:
                                counts[k_idx] -= 1
                            
                            active[t_add] = True
                            for k_idx in triple_to_k222s[t_add]:
                                counts[k_idx] += 1
                            
                            improved = True
                            found_2for1 = True
                            break
                        
                        # Try removing r1 and r2 together
                        for r2 in active_indices[:50]:
                            if r2 == r1:
                                continue
                            if time_left() < 0.5:
                                return active, counts, current_size
                            
                            # Check if removing both r1 and r2 fixes all blockers
                            all_fixed = True
                            for k_idx in blockers:
                                if r1 in k222_triples[k_idx] or r2 in k222_triples[k_idx]:
                                    continue
                                all_fixed = False
                                break
                            
                            if all_fixed:
                                # Perform the 2-for-1 move
                                active[r1] = False
                                for k_idx in triple_to_k222s[r1]:
                                    counts[k_idx] -= 1
                                active[r2] = False
                                for k_idx in triple_to_k222s[r2]:
                                    counts[k_idx] -= 1
                                
                                active[t_add] = True
                                for k_idx in triple_to_k222s[t_add]:
                                    counts[k_idx] += 1
                                
                                improved = True
                                found_2for1 = True
                                break
                            
                            if found_2for1:
                                break
                        
                        if found_2for1:
                            break
                    
                    if found_2for1:
                        break
                
                if not found_2for1:
                    break
        
        return active, counts, current_size
    
    # Multi-round with perturbations
    best_size = 0
    best_active = [False] * N_TRIPLES
    best_counts = [0] * N_K222
    
    seed = 42
    while time_left() > 10.0:
        # Greedy initialization
        active, counts, current_size = greedy_init(seed)
        
        # Local search
        active, counts, current_size = local_search(active, counts, current_size)
        
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]
        
        # Multiple perturbation strategies
        if time_left() > 15.0 and best_size > 0:
            # Strategy 1: Remove random triples
            perturbed_active = best_active[:]
            perturbed_counts = best_counts[:]
            perturbed_size = best_size
            
            num_remove = random.randint(3, 20)
            active_list = [i for i in range(N_TRIPLES) if perturbed_active[i]]
            to_remove = random.sample(active_list, min(num_remove, len(active_list)))
            
            for i in to_remove:
                perturbed_active[i] = False
                perturbed_size -= 1
                for k_idx in triple_to_k222s[i]:
                    perturbed_counts[k_idx] -= 1
            
            perturbed_active, perturbed_counts, perturbed_size = local_search(
                perturbed_active, perturbed_counts, perturbed_size
            )
            
            if perturbed_size > best_size:
                best_size = perturbed_size
                best_active = perturbed_active[:]
                best_counts = perturbed_counts[:]
        
        # Strategy 2: Swap-based perturbation - replace random active triples with random inactive
        if time_left() > 15.0 and best_size > 0:
            perturbed_active = best_active[:]
            perturbed_counts = best_counts[:]
            perturbed_size = best_size
            
            active_list = [i for i in range(N_TRIPLES) if perturbed_active[i]]
            inactive_list = [i for i in range(N_TRIPLES) if not perturbed_active[i]]
            
            if active_list and inactive_list:
                num_swap = random.randint(2, 8)
                to_remove = random.sample(active_list, min(num_swap, len(active_list)))
                to_add = random.sample(inactive_list, min(num_swap, len(inactive_list)))
                
                for i in to_remove:
                    perturbed_active[i] = False
                    perturbed_size -= 1
                    for k_idx in triple_to_k222s[i]:
                        perturbed_counts[k_idx] -= 1
                
                for i in to_add:
                    # Check if addable
                    addable = True
                    for k_idx in triple_to_k222s[i]:
                        if perturbed_counts[k_idx] >= 7:
                            addable = False
                            break
                    if addable:
                        perturbed_active[i] = True
                        perturbed_size += 1
                        for k_idx in triple_to_k222s[i]:
                            perturbed_counts[k_idx] += 1
                
                perturbed_active, perturbed_counts, perturbed_size = local_search(
                    perturbed_active, perturbed_counts, perturbed_size
                )
                
                if perturbed_size > best_size:
                    best_size = perturbed_size
                    best_active = perturbed_active[:]
                    best_counts = perturbed_counts[:]
        
        seed += 1
    
    # Final polish: try to add any remaining addable triples
    if best_size > 0 and time_left() > 5.0:
        final_active = best_active[:]
        final_counts = best_counts[:]
        final_size = best_size
        final_active, final_counts, final_size = local_search(
            final_active, final_counts, final_size, max_iterations=100000
        )
        if final_size > best_size:
            best_size = final_size
            best_active = final_active[:]
            best_counts = final_counts[:]
    
    # Construct result
    result = []
    for i in range(N_TRIPLES):
        if best_active[i]:
            result.append(list(ALL_TRIPLES[i]))
    
    return result

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
