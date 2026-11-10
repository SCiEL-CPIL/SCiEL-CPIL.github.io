import time
import random

def construct():
    """
    Construct a 4-term-progression-free subset of [1, 512].
    Uses a stochastic local search (WalkSAT-style) on the Max-SAT formulation.
    Variables: x in [1, 512] (1 if in set).
    Clauses: For each 4-AP (a,b,c,d), clause is NOT (a AND b AND c AND d).
    We search for a satisfying assignment (0 conflicts) with maximum number of 1s.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 512
    
    # Precompute all 4-APs and their masks
    # aps[x] will be a list of tuples representing the other 3 elements in each 4-AP containing x
    # Actually, for the "list of violated APs" approach, it's better to store the APs themselves.
    
    # Let's store all 4-APs as a list of tuples (a, b, c, d)
    all_aps = []
    # Map from element to list of indices of APs containing it
    elem_to_ap_indices = [[] for _ in range(n + 1)]
    
    ap_id = 0
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            all_aps.append((a, b, c, e))
            elem_to_ap_indices[a].append(ap_id)
            elem_to_ap_indices[b].append(ap_id)
            elem_to_ap_indices[c].append(ap_id)
            elem_to_ap_indices[e].append(ap_id)
            ap_id += 1
            
    num_aps = len(all_aps)
    
    # Helper to check if an AP is satisfied (i.e., NOT all 4 are in S)
    # We will maintain a boolean array `ap_violated` of size num_aps
    # and a count `conflict_count`
    
    def greedy_init():
        """Randomized greedy initialization."""
        S = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # For efficiency in greedy, we can just check conflicts on the fly
        # Since n=512, O(n^2) or O(n^3) is fine for init.
        for x in candidates:
            valid = True
            # Check if adding x creates a 4-AP
            for ap_idx in elem_to_ap_indices[x]:
                a, b, c, e = all_aps[ap_idx]
                # Check if the other 3 are in S
                others = [y for y in (a, b, c, e) if y != x]
                if all(o in S for o in others):
                    valid = False
                    break
            if valid:
                S.add(x)
        return S

    def local_search(initial_S):
        """
        WalkSAT-style local search.
        State: S (set of elements), violated_aps (list of indices of violated APs).
        """
        S = set(initial_S)
        S_list = list(S)
        
        # Initialize violated_aps
        # Since initial_S is feasible, violated_aps should be empty
        violated_aps = []
        
        best_S = set(S)
        best_size = len(S)
        
        stagnation = 0
        
        while time.time() < deadline:
            if len(violated_aps) == 0:
                # Feasible state
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                
                # Try to add an element
                # Pick a random element not in S
                not_in_S = [x for x in range(1, n + 1) if x not in S]
                if not not_in_S:
                    break # Maxed out?
                
                x = random.choice(not_in_S)
                
                # Check conflicts if we add x
                new_violations = []
                for ap_idx in elem_to_ap_indices[x]:
                    a, b, c, e = all_aps[ap_idx]
                    others = [y for y in (a, b, c, e) if y != x]
                    if all(o in S for o in others):
                        new_violations.append(ap_idx)
                
                if not new_violations:
                    # Safe move
                    S.add(x)
                    stagnation = 0
                else:
                    # Infeasible move. Decide whether to take it.
                    # Heuristic: Take it if we are stagnating or with some probability
                    # to escape local optima.
                    if stagnation > 50 or random.random() < 0.1:
                        S.add(x)
                        violated_aps.extend(new_violations)
                        stagnation += 1
                    else:
                        stagnation += 1
            else:
                # Infeasible state. Must repair.
                # Pick a random violated AP
                ap_idx = random.choice(violated_aps)
                a, b, c, e = all_aps[ap_idx]
                
                # Pick a random element from this AP to remove
                # To be more effective, we might want to remove an element that is in many violations?
                # For now, random is fine.
                y = random.choice([a, b, c, e])
                
                # Remove y from S
                if y in S:
                    S.remove(y)
                    
                    # Update violated_aps
                    # Remove all APs from violated_aps that contain y
                    # Also, removing y might *break* other APs? No, removing an element only breaks APs.
                    # So we just need to filter out APs containing y.
                    # But we must be careful: if an AP was violated *only* because of y, it is now fixed.
                    # If an AP was violated for other reasons, it remains violated.
                    # Our `violated_aps` list contains indices of APs that are *currently* fully in S.
                    # If we remove y, any AP containing y is no longer fully in S (since y is gone).
                    # So we can simply remove all indices from violated_aps that correspond to APs containing y.
                    
                    # Optimization: Precompute set of AP indices containing each y?
                    # We have elem_to_ap_indices.
                    # We can create a set of indices to remove.
                    indices_to_remove = set(elem_to_ap_indices[y])
                    
                    if indices_to_remove:
                        violated_aps = [idx for idx in violated_aps if idx not in indices_to_remove]
                        
                # If we removed y, we might have fixed multiple APs.
                # The loop will continue to repair any remaining violations.
                
                # Reset stagnation on repair? Or keep it?
                # Let's keep it to encourage staying in feasible region if we return to it.
                # But if we oscillate, we might get stuck.
                # Let's add a small probability to reset stagnation on successful repair (violated_aps becomes empty).
                if not violated_aps:
                    stagnation = 0

        return best_S

    # Run multiple restarts
    best_overall = set()
    
    for i in range(50): # Max restarts, but limited by time
        if time.time() > deadline - 5:
            break
            
        S_init = greedy_init()
        S_res = local_search(S_init)
        
        if len(S_res) > len(best_overall):
            best_overall = S_res
            
        # Early stop if we found a very large set? 
        # Theoretical max is around n/3? No, 4-AP free sets can be denser.
        # Salem-Spencer is ~ n / (log n)^c.
        # For n=512, let's see what we get.
        
    return sorted(best_overall)

# Verification helper (not used in construct, but good for debug)
def is_4ap_free(S):
    S_set = set(S)
    n = 512
    for a in S:
        for d in range(1, (n - a) // 3 + 1):
            if (a + d) in S_set and (a + 2*d) in S_set and (a + 3*d) in S_set:
                return False
    return True
