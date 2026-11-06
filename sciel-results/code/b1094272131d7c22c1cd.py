"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_conflicts_bitmask():
    """Build conflict structure using bitmasks for fast checking.
    Returns:
        conflict_mask[x]: integer bitmask where bit y is set if {x,y} is in some bad triple
        bad_triples: list of (a,b,c) tuples that are bad
    """
    # For each x, we need to know: for which pairs (y,z) does {x,y,z} form a bad triple?
    # Instead of storing pairs, we store: for each x, a list of (y, z) pairs
    # But for fast checking, we can precompute for each x:
    #   conflict_pairs[x] = list of (y, z) such that {x,y,z} is bad
    # Then _can_add(x, S) checks: for each (y,z) in conflict_pairs[x], is y in S AND z in S?
    
    # Better: use bitmasks
    # mask[x] = bitmask of elements y such that there exists z with {x,y,z} bad
    # But that's not quite right either. We need to check if BOTH y and z are in S.
    
    # Let's use: for each x, store a list of bitmasks where each bitmask represents a pair (y,z)
    # Then _can_add(x, S_mask) checks: for each pair_mask in conflict_masks[x], 
    #   is (S_mask & pair_mask) == pair_mask?
    
    conflict_masks = {x: [] for x in range(1, 129)}
    nums = list(range(1, 129))
    
    for a, b, c in combinations(nums, 3):
        lab = _lcm(a, b)
        lac = _lcm(a, c)
        lbc = _lcm(b, c)
        if lab == lac == lbc:
            # For a: the conflicting pair is (b, c)
            pair_mask = (1 << b) | (1 << c)
            conflict_masks[a].append(pair_mask)
            # For b: the conflicting pair is (a, c)
            pair_mask = (1 << a) | (1 << c)
            conflict_masks[b].append(pair_mask)
            # For c: the conflicting pair is (a, b)
            pair_mask = (1 << a) | (1 << b)
            conflict_masks[c].append(pair_mask)
    
    return conflict_masks

def _can_add_fast(x, selected_mask, conflict_masks):
    """Fast check using bitmask: can x be added to selected_mask?"""
    for pair_mask in conflict_masks[x]:
        if (selected_mask & pair_mask) == pair_mask:
            return False
    return True

def _greedy_min_conflict(conflict_masks, deadline):
    """Greedy that at each step picks the element with fewest conflicts with current selection."""
    selected_mask = 0
    selected = []
    available = set(range(1, 129))
    
    while available and time.time() < deadline:
        # Score each available element by how many conflicts it would have
        best_x = None
        best_score = float('inf')
        
        for x in available:
            # Count how many pairs in conflict_masks[x] are both in selected_mask
            conflict_count = 0
            for pair_mask in conflict_masks[x]:
                if (selected_mask & pair_mask) == pair_mask:
                    conflict_count += 1
                    break  # Already conflicts, no need to count more
            if conflict_count == 0:
                # No conflict - pick this one (prefer smaller index for determinism)
                if best_x is None or x < best_x:
                    best_x = x
                    best_score = 0
            else:
                # This element conflicts, skip
                pass
        
        if best_x is None:
            # All available elements conflict - try to find one that conflicts least
            # Actually, if all conflict, we can't add any more
            break
        
        # Add best_x
        selected_mask |= (1 << best_x)
        selected.append(best_x)
        available.discard(best_x)
        
        # Remove elements that now conflict with the current selection
        # Actually, we should keep them available but they'll be rejected in next iteration
        # For efficiency, we could prune, but let's keep it simple
    
    return selected, selected_mask

def _greedy_order(conflict_masks, order, deadline):
    """Greedy with a fixed order."""
    selected_mask = 0
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_fast(x, selected_mask, conflict_masks):
            selected_mask |= (1 << x)
            selected.append(x)
    return selected, selected_mask

def _local_search_fast(selected_mask, conflict_masks, time_limit=10):
    """Local search using bitmask operations."""
    best_mask = selected_mask
    best_size = bin(selected_mask).count('1')
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        # Try swap: remove one, add one
        for rem in range(1, 129):
            if not (selected_mask & (1 << rem)):
                continue
            candidate_mask = selected_mask & ~(1 << rem)
            for x in range(1, 129):
                if candidate_mask & (1 << x):
                    continue
                if _can_add_fast(x, candidate_mask, conflict_masks):
                    new_mask = candidate_mask | (1 << x)
                    new_size = bin(new_mask).count('1')
                    if new_size > best_size:
                        best_mask = new_mask
                        best_size = new_size
                        selected_mask = new_mask
                        improved = True
                        break
            if improved:
                break
        if not improved:
            # Try 2-for-1: remove two, add one
            improved = False
            for rem1 in range(1, 129):
                if not (selected_mask & (1 << rem1)):
                    continue
                for rem2 in range(rem1 + 1, 129):
                    if not (selected_mask & (1 << rem2)):
                        continue
                    candidate_mask = selected_mask & ~(1 << rem1) & ~(1 << rem2)
                    for x in range(1, 129):
                        if candidate_mask & (1 << x):
                            continue
                        if _can_add_fast(x, candidate_mask, conflict_masks):
                            new_mask = candidate_mask | (1 << x)
                            new_size = bin(new_mask).count('1')
                            if new_size > best_size:
                                best_mask = new_mask
                                best_size = new_size
                                selected_mask = new_mask
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Convert back to list
    result = [x for x in range(1, 129) if best_mask & (1 << x)]
    return result

def _anneal_fast(selected_mask, conflict_masks, time_limit=15):
    """Simulated annealing using bitmask operations."""
    current_mask = selected_mask
    best_mask = selected_mask
    best_size = bin(selected_mask).count('1')
    start = time.time()
    temp = 5.0
    cooling = 0.97
    
    while time.time() - start < time_limit and temp > 0.1:
        temp *= cooling
        
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Swap: remove one, add one
            selected_list = [x for x in range(1, 129) if current_mask & (1 << x)]
            if not selected_list:
                continue
            rem = random.choice(selected_list)
            candidate_mask = current_mask & ~(1 << rem)
            
            # Try to add a random element
            available = [x for x in range(1, 129) if not (candidate_mask & (1 << x))]
            if not available:
                continue
            add = random.choice(available)
            
            if _can_add_fast(add, candidate_mask, conflict_masks):
                new_mask = candidate_mask | (1 << add)
            else:
                new_mask = candidate_mask
        elif move_type < 0.8:
            # Remove one, try to add greedily
            selected_list = [x for x in range(1, 129) if current_mask & (1 << x)]
            if not selected_list:
                continue
            rem = random.choice(selected_list)
            candidate_mask = current_mask & ~(1 << rem)
            
            new_mask = candidate_mask
            for x in range(1, 129):
                if not (new_mask & (1 << x)) and _can_add_fast(x, new_mask, conflict_masks):
                    new_mask |= (1 << x)
        else:
            # Remove two, try to add greedily
            selected_list = [x for x in range(1, 129) if current_mask & (1 << x)]
            if len(selected_list) < 2:
                continue
            rem1, rem2 = random.sample(selected_list, 2)
            candidate_mask = current_mask & ~(1 << rem1) & ~(1 << rem2)
            
            new_mask = candidate_mask
            for x in range(1, 129):
                if not (new_mask & (1 << x)) and _can_add_fast(x, new_mask, conflict_masks):
                    new_mask |= (1 << x)
        
        new_size = bin(new_mask).count('1')
        if new_size > best_size:
            best_mask = new_mask
            best_size = new_size
            current_mask = new_mask
        elif random.random() < temp / 10.0:
            current_mask = new_mask
    
    result = [x for x in range(1, 129) if best_mask & (1 << x)]
    return result

def _branch_and_bound(conflict_masks, initial_mask, deadline):
    """
    Branch and bound to find maximum independent set in the 3-uniform hypergraph.
    State: (current_mask, available_mask, forbidden_pairs)
    """
    best_size = bin(initial_mask).count('1')
    best_mask = initial_mask
    
    # Precompute conflict_with[v][w] = bitmask of u such that {v, u, w} is bad
    # This allows fast removal of vertices from 'available' when adding v
    conflict_with = {}
    for v in range(1, 129):
        conflict_with[v] = {}
        for w in range(1, 129):
            if v == w:
                continue
            mask = 0
            for pair_mask in conflict_masks[v]:
                # pair_mask represents {u, z} such that {v, u, z} is bad
                # We need to find u such that w is in {u, z}
                # This logic is a bit tricky. Let's precompute triples directly.
                pass
    
    # Better: Precompute a dictionary mapping (v, w) to bitmask of u
    # such that {v, u, w} is a bad triple.
    triple_conflicts = {}
    nums = list(range(1, 129))
    # Re-iterate triples to build this
    # We need to rebuild triples or extract from conflict_masks
    # conflict_masks[v] has pairs {u, z}.
    # So for a fixed v, and a candidate w, if w is in pair {u, z}, then u is the conflict.
    # We can build this map.
    for v in range(1, 129):
        for w in range(1, 129):
            if v == w:
                continue
            mask = 0
            for pair_mask in conflict_masks[v]:
                # Check if w is in pair_mask
                if pair_mask & (1 << w):
                    # The other element is the conflict
                    # pair_mask has exactly 2 bits set
                    other = pair_mask ^ (1 << w)
                    if other:
                        mask |= other
            triple_conflicts[(v, w)] = mask

    def popcount(x):
        return bin(x).count('1')

    def solve(curr_mask, avail_mask, forbidden_pairs, depth=0):
        nonlocal best_size, best_mask
        
        # Pruning: if time is up, return
        if time.time() > deadline:
            return
            
        # Upper bound
        # We can add at most popcount(avail_mask) more vertices.
        # But we have forbidden pairs.
        # Simple bound: popcount(curr_mask) + popcount(avail_mask)
        if popcount(curr_mask) + popcount(avail_mask) <= best_size:
            return
            
        if not avail_mask:
            if popcount(curr_mask) > best_size:
                best_size = popcount(curr_mask)
                best_mask = curr_mask
            return
            
        # Choose a vertex to branch on
        # Heuristic: Pick vertex with highest degree in the induced subgraph?
        # Or just pick the lowest available vertex for determinism and simplicity.
        # To improve, we could pick the one that conflicts most with others.
        # For now, let's pick the lowest bit.
        # This might be slow if the tree is wide.
        
        # Better branching variable selection:
        # Pick v from avail_mask such that it has the most conflicts in avail_mask?
        # Or just iterate.
        
        # Let's pick the first available vertex
        # Find lowest set bit
        v = (avail_mask & -avail_mask).bit_length() - 1
        if v == 0:
             # If v=0, it means avail_mask was 0, but we checked that.
             # bit_length of 1<<0 is 1, -1 is 0.
             # Wait, numbers are 1..128.
             # If avail_mask has bit 1 set (value 2), bit_length is 2, -1 is 1.
             # If avail_mask has bit 128 set (value 1<<128), bit_length is 129, -1 is 128.
             pass
             
        # v is the index of the lowest set bit in avail_mask
        # Note: Python integers are arbitrary precision, so this works.
        
        # Branch 1: Include v
        new_curr = curr_mask | (1 << v)
        new_avail = avail_mask & ~(1 << v)
        
        # Remove vertices from new_avail that conflict with v given curr_mask
        # If we add v, any u in new_avail such that there exists w in curr_mask
        # with {v, u, w} bad must be removed.
        removal_mask = 0
        # Iterate through w in curr_mask
        temp_mask = curr_mask
        while temp_mask:
            w = (temp_mask & -temp_mask).bit_length() - 1
            removal_mask |= triple_conflicts.get((v, w), 0)
            temp_mask &= temp_mask - 1
        
        new_avail &= ~removal_mask
        
        # Add new forbidden pairs from conflict_masks[v]
        new_forbidden = forbidden_pairs + conflict_masks[v]
        
        # Optimization: If any forbidden pair is fully contained in new_avail, 
        # we can't add both. This doesn't change the mask, but affects the bound.
        # For now, we just recurse.
        
        solve(new_curr, new_avail, new_forbidden, depth + 1)
        
        # Branch 2: Exclude v
        new_avail_excl = avail_mask & ~(1 << v)
        solve(curr_mask, new_avail_excl, forbidden_pairs, depth + 1)

    solve(initial_mask, 0, []) # Wait, initial_mask is the START, avail should be all others?
    # The function signature in solve assumes curr_mask is the set of INCLUDED vertices.
    # And avail_mask is the set of vertices that CAN be added.
    # So initially, curr_mask = initial_mask (from greedy), avail_mask = all vertices NOT in initial_mask.
    
    all_mask = (1 << 129) - 2 # bits 1 to 128
    # Actually, 1<<129 is bit 129. We want bits 1..128.
    # Sum of 1<<i for i in 1..128.
    # (1 << 129) - 2 gives bits 1 to 128.
    
    initial_avail = all_mask & ~initial_mask
    
    solve(initial_mask, initial_avail, [])
    
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Precompute conflict structure using bitmasks
    conflict_masks = _build_conflicts_bitmask()
    
    best = []
    best_size = 0
    best_mask = 0
    
    # Strategy 1: Min-conflict greedy
    sel, sel_mask = _greedy_min_conflict(conflict_masks, deadline - 60)
    best = sel
    best_size = len(sel)
    best_mask = sel_mask
    
    # Strategy 2: Greedy with natural order
    if time.time() < deadline - 55:
        sel, sel_mask = _greedy_order(conflict_masks, list(range(1, 129)), deadline - 55)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
            best_mask = sel_mask
    
    # Strategy 3: Greedy with reverse order
    if time.time() < deadline - 50:
        sel, sel_mask = _greedy_order(conflict_masks, list(range(128, 0, -1)), deadline - 50)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
            best_mask = sel_mask
    
    # Strategy 4: Local search on best
    if time.time() < deadline - 30:
        improved = _local_search_fast(best_mask, conflict_masks, time_limit=10)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
            best_mask = sum(1 << x for x in best)
    
    # Strategy 5: Multiple random restarts
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 20 and restarts < 500:
        restarts += 1
        order = list(range(1, 129))
        random.shuffle(order)
        
        sel, sel_mask = _greedy_order(conflict_masks, order, deadline - 20)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
            best_mask = sel_mask
        
        # Improve if close to best
        if len(sel) >= best_size - 2 and time.time() < deadline - 25:
            improved = _local_search_fast(sel_mask, conflict_masks, time_limit=2)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
                best_mask = sum(1 << x for x in best)
    
    # Strategy 6: Simulated annealing on best solution
    if time.time() < deadline - 10 and best_size > 0:
        improved = _anneal_fast(best_mask, conflict_masks, time_limit=10)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
            best_mask = sum(1 << x for x in best)
    
    # Strategy 7: Branch and Bound
    if time.time() < deadline - 5 and best_size > 0:
        improved_mask = _branch_and_bound(conflict_masks, best_mask, deadline - 2)
        improved = [x for x in range(1, 129) if improved_mask & (1 << x)]
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
            best_mask = improved_mask
    
    # Strategy 8: Final local search
    if time.time() < deadline - 3 and best_size > 0:
        improved = _local_search_fast(best_mask, conflict_masks, time_limit=3)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    if not best:
        return [1]
    
    return best
# EVOLVE-BLOCK-END
