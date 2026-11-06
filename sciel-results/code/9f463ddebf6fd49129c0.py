import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    
    # Precompute 3-APs and 4-APs
    # Element i (1-indexed) corresponds to index (i-1) in arrays
    
    aps3 = []
    aps4 = []
    
    elem_to_aps3 = [[] for _ in range(n)]
    elem_to_aps4 = [[] for _ in range(n)]
    
    # Generate 3-APs
    ap_id = 0
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                # Store as 0-indexed tuple
                aps3.append((a-1, b-1, c-1))
                elem_to_aps3[a-1].append(ap_id)
                elem_to_aps3[b-1].append(ap_id)
                elem_to_aps3[c-1].append(ap_id)
                ap_id += 1
                
    # Generate 4-APs
    ap_id = 0
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                aps4.append((a-1, b-1, c-1, e-1))
                elem_to_aps4[a-1].append(ap_id)
                elem_to_aps4[b-1].append(ap_id)
                elem_to_aps4[c-1].append(ap_id)
                elem_to_aps4[e-1].append(ap_id)
                ap_id += 1

    def count_3aps(A):
        count = 0
        for a, b, c in aps3:
            if A[a] and A[b] and A[c]:
                count += 1
        return count
        
    def count_4aps(A):
        count = 0
        for a, b, c, d in aps4:
            if A[a] and A[b] and A[c] and A[d]:
                count += 1
        return count

    # Helper to calculate gain of adding v to set A
    # Returns number of new 3-APs formed
    def gain_add(A, v):
        gain = 0
        for ap_idx in elem_to_aps3[v]:
            a, b, c = aps3[ap_idx]
            # Check if other two are in A
            if a == v:
                if A[b] and A[c]: gain += 1
            elif b == v:
                if A[a] and A[c]: gain += 1
            else: # c == v
                if A[a] and A[b]: gain += 1
        return gain

    # Helper to calculate loss of removing u from set A
    # Returns number of 3-APs destroyed
    def loss_remove(A, u):
        loss = 0
        for ap_idx in elem_to_aps3[u]:
            a, b, c = aps3[ap_idx]
            if a == u:
                if A[b] and A[c]: loss += 1
            elif b == u:
                if A[a] and A[c]: loss += 1
            else: # c == u
                if A[a] and A[b]: loss += 1
        return loss

    def find_conflicts(A, v):
        """Find 4-APs that would be completed by adding v to A."""
        conflicts = []
        for ap_idx in elem_to_aps4[v]:
            a, b, c, d = aps4[ap_idx]
            # Check if other 3 are in A
            if a == v:
                if A[b] and A[c] and A[d]:
                    conflicts.append(ap_idx)
            elif b == v:
                if A[a] and A[c] and A[d]:
                    conflicts.append(ap_idx)
            elif c == v:
                if A[a] and A[b] and A[d]:
                    conflicts.append(ap_idx)
            else: # d == v
                if A[a] and A[b] and A[c]:
                    conflicts.append(ap_idx)
        return conflicts

    def run_sa(seed):
        rng = random.Random(seed)
        
        # Initialize with a random valid set
        # Use a simple greedy to get a valid start, or just random and repair
        A = np.zeros(n, dtype=bool)
        
        # Greedy initialization for speed and validity
        candidates = list(range(n))
        rng.shuffle(candidates)
        for x in candidates:
            # Check if adding x creates 4-AP
            valid = True
            for ap_idx in elem_to_aps4[x]:
                a, b, c, d = aps4[ap_idx]
                if a == x:
                    if A[b] and A[c] and A[d]:
                        valid = False
                        break
                elif b == x:
                    if A[a] and A[c] and A[d]:
                        valid = False
                        break
                elif c == x:
                    if A[a] and A[b] and A[d]:
                        valid = False
                        break
                else:
                    if A[a] and A[b] and A[c]:
                        valid = False
                        break
            if valid:
                A[x] = True
                
        current_score = count_3aps(A)
        best_score = current_score
        best_A = A.copy()
        
        # Simulated Annealing parameters
        T = 10.0
        T_min = 0.1
        cooling_rate = 0.999
        
        while time.time() < deadline - 1.0:
            if T < T_min:
                break
                
            # Pick a random move type: Add, Remove, or Swap
            # Swaps are more effective for maintaining cardinality
            move_type = rng.choice(['add', 'remove', 'swap'])
            
            delta = 0
            new_A = A.copy()
            repair_removed = -1
            
            if move_type == 'add':
                # Pick random v not in A
                empty = np.where(~A)[0]
                if len(empty) == 0: continue
                v = empty[rng.randint(0, len(empty)-1)]
                
                gain = gain_add(A, v)
                conflicts = find_conflicts(A, v)
                
                if conflicts:
                    # Need to repair
                    # Pick a random conflict
                    c_idx = conflicts[rng.randint(0, len(conflicts)-1)]
                    a, b, c, d = aps4[c_idx]
                    # Elements in A involved in this 4-AP (excluding v)
                    others = [x for x in (a,b,c,d) if x != v]
                    # Pick one to remove
                    u = others[rng.randint(0, len(others)-1)]
                    
                    # Check if u is in A (it should be)
                    if not A[u]: continue
                    
                    loss = loss_remove(A, u)
                    # Net delta: gain from v, loss from u
                    # Note: removing u might break other 3-APs, adding v might create others.
                    # This approximation is valid for local step.
                    delta = gain - loss
                    
                    A[v] = True
                    A[u] = False
                    repair_removed = u
                else:
                    delta = gain
                    A[v] = True
                    
            elif move_type == 'remove':
                # Pick random u in A
                full = np.where(A)[0]
                if len(full) == 0: continue
                u = full[rng.randint(0, len(full)-1)]
                
                loss = loss_remove(A, u)
                delta = -loss
                A[u] = False
                
            else: # swap
                # Pick random u in A, v not in A
                full = np.where(A)[0]
                empty = np.where(~A)[0]
                if len(full) == 0 or len(empty) == 0: continue
                
                u = full[rng.randint(0, len(full)-1)]
                v = empty[rng.randint(0, len(empty)-1)]
                
                if u == v: continue
                
                # Calculate delta
                # 1. Loss from removing u
                loss_u = loss_remove(A, u)
                
                # 2. Gain from adding v
                # We need to calculate gain in the context of A without u
                # This is tricky. 
                # Approximation: Calculate gain_add(A, v) and subtract any 3-APs that relied on u.
                # A 3-AP is formed if (A\{u}) contains the other 2 elements.
                gain_v = 0
                for ap_idx in elem_to_aps3[v]:
                    a, b, c = aps3[ap_idx]
                    if a == v:
                        # Need b, c in A and not u
                        if A[b] and A[c] and b != u and c != u:
                            gain_v += 1
                    elif b == v:
                        if A[a] and A[c] and a != u and c != u:
                            gain_v += 1
                    else:
                        if A[a] and A[b] and a != u and b != u:
                            gain_v += 1
                
                delta = gain_v - loss_u
                
                # Check 4-AP validity for swap
                # Remove u, Add v.
                # Check if v completes a 4-AP with elements in A\{u}
                valid = True
                for ap_idx in elem_to_aps4[v]:
                    a, b, c, d = aps4[ap_idx]
                    if a == v:
                        if A[b] and A[c] and A[d] and b != u and c != u and d != u:
                            valid = False
                            break
                    elif b == v:
                        if A[a] and A[c] and A[d] and a != u and c != u and d != u:
                            valid = False
                            break
                    elif c == v:
                        if A[a] and A[b] and A[d] and a != u and b != u and d != u:
                            valid = False
                            break
                    else:
                        if A[a] and A[b] and A[c] and a != u and b != u and c != u:
                            valid = False
                            break
                
                if not valid:
                    # Repair: pick one of the conflicting elements to remove instead of u?
                    # Or just reject. Repairing swaps is complex. Let's reject for now.
                    continue
                
                A[u] = False
                A[v] = True
            
            # Acceptance
            if delta > 0:
                accept = True
            else:
                # Metropolis criterion
                # Use exp(delta / T)
                # delta is negative
                try:
                    prob = math.exp(delta / T)
                except OverflowError:
                    prob = 0.0
                accept = (rng.random() < prob)
            
            if accept:
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_A = A.copy()
            else:
                # Revert A
                # Since we modified A in place, we need to revert specific changes
                # This is error-prone. Let's use a copy for the trial.
                # Optimization: Only copy if we might accept? No, we need to know delta first.
                # The current code modifies A directly. We must revert.
                
                if move_type == 'add':
                    if repair_removed != -1:
                        # v was added, u was removed
                        A[repair_removed] = True
                        # v was the one added, find it? 
                        # We didn't store v. 
                        # This implementation is flawed for reverting.
                        pass
                    else:
                        # v was added. We don't know v.
                        pass
                elif move_type == 'remove':
                    # u was removed. We don't know u.
                    pass
                else:
                    # u removed, v added. We don't know u, v.
                    pass
                
                # To fix this, we should store the changes.
                # Let's refactor the loop to store changes.
                
            T *= cooling_rate

        return best_A, best_score

    # Main logic
    best_mask = 0
    best_set = set()
    best_score = 0
    
    # Run multiple SA instances
    # Since SA is stochastic, running a few in parallel (sequentially) helps.
    # But SA is slow in Python. Let's do 2-3 runs.
    
    num_runs = 3
    time_per_run = (deadline - time.time()) / num_runs
    
    for i in range(num_runs):
        if time.time() > deadline - 5:
            break
        
        # We need to fix the revert logic in run_sa.
        # Let's rewrite run_sa with proper state management.
        
        def run_sa_fixed(seed, time_limit):
            rng = random.Random(seed)
            A = np.zeros(n, dtype=bool)
            
            # Greedy Init
            candidates = list(range(n))
            rng.shuffle(candidates)
            for x in candidates:
                valid = True
                for ap_idx in elem_to_aps4[x]:
                    a, b, c, d = aps4[ap_idx]
                    if a == x:
                        if A[b] and A[c] and A[d]:
                            valid = False; break
                    elif b == x:
                        if A[a] and A[c] and A[d]:
                            valid = False; break
                    elif c == x:
                        if A[a] and A[b] and A[d]:
                            valid = False; break
                    else:
                        if A[a] and A[b] and A[c]:
                            valid = False; break
                if valid:
                    A[x] = True
                    
            current_score = count_3aps(A)
            best_score = current_score
            best_A = A.copy()
            
            T = 10.0
            T_min = 0.1
            cooling_rate = 0.9995 # Slower cooling for better exploration
            
            while time.time() < time_limit:
                if T < T_min:
                    break
                    
                move_type = rng.choice(['add', 'remove', 'swap'])
                delta = 0
                changes = [] # List of (index, new_val) to revert
                
                if move_type == 'add':
                    empty = np.where(~A)[0]
                    if len(empty) == 0: continue
                    v = empty[rng.randint(0, len(empty)-1)]
                    
                    gain = gain_add(A, v)
                    conflicts = find_conflicts(A, v)
                    
                    if conflicts:
                        c_idx = conflicts[rng.randint(0, len(conflicts)-1)]
                        a, b, c, d = aps4[c_idx]
                        others = [x for x in (a,b,c,d) if x != v]
                        u = others[rng.randint(0, len(others)-1)]
                        
                        if not A[u]: continue
                        
                        loss = loss_remove(A, u)
                        delta = gain - loss
                        
                        A[v] = True
                        changes.append((v, True))
                        A[u] = False
                        changes.append((u, False))
                    else:
                        delta = gain
                        A[v] = True
                        changes.append((v, True))
                        
                elif move_type == 'remove':
                    full = np.where(A)[0]
                    if len(full) == 0: continue
                    u = full[rng.randint(0, len(full)-1)]
                    
                    loss = loss_remove(A, u)
                    delta = -loss
                    A[u] = False
                    changes.append((u, False))
                    
                else: # swap
                    full = np.where(A)[0]
                    empty = np.where(~A)[0]
                    if len(full) == 0 or len(empty) == 0: continue
                    
                    u = full[rng.randint(0, len(full)-1)]
                    v = empty[rng.randint(0, len(empty)-1)]
                    
                    if u == v: continue
                    
                    loss_u = loss_remove(A, u)
                    
                    gain_v = 0
                    for ap_idx in elem_to_aps3[v]:
                        a, b, c = aps3[ap_idx]
                        if a == v:
                            if A[b] and A[c] and b != u and c != u:
                                gain_v += 1
                        elif b == v:
                            if A[a] and A[c] and a != u and c != u:
                                gain_v += 1
                        else:
                            if A[a] and A[b] and a != u and b != u:
                                gain_v += 1
                                
                    delta = gain_v - loss_u
                    
                    valid = True
                    for ap_idx in elem_to_aps4[v]:
                        a, b, c, d = aps4[ap_idx]
                        if a == v:
                            if A[b] and A[c] and A[d] and b != u and c != u and d != u:
                                valid = False; break
                        elif b == v:
                            if A[a] and A[c] and A[d] and a != u and c != u and d != u:
                                valid = False; break
                        elif c == v:
                            if A[a] and A[b] and A[d] and a != u and b != u and d != u:
                                valid = False; break
                        else:
                            if A[a] and A[b] and A[c] and a != u and b != u and c != u:
                                valid = False; break
                                
                    if not valid:
                        continue
                    
                    A[u] = False
                    changes.append((u, False))
                    A[v] = True
                    changes.append((v, True))
                    
                if delta > 0:
                    accept = True
                else:
                    try:
                        prob = math.exp(delta / T)
                    except OverflowError:
                        prob = 0.0
                    accept = (rng.random() < prob)
                
                if accept:
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_A = A.copy()
                else:
                    # Revert
                    for idx, val in changes:
                        A[idx] = not val # Wait, val is the NEW value. Old value is NOT val.
                        # Actually, A[idx] was flipped. So A[idx] = not A[idx] reverts it.
                        # But changes stores (idx, new_val). 
                        # If new_val was True, old was False. A[idx] is now True. Revert to False.
                        # If new_val was False, old was True. A[idx] is now False. Revert to True.
                        # So A[idx] = not val is correct IF val is the state we set.
                        # Let's just use A[idx] = not A[idx] since we only flip bits.
                        A[idx] = not A[idx]
                        
                T *= cooling_rate

            return best_A, best_score

        A, score = run_sa_fixed(i, time.time() + time_per_run - 1)
        
        if score > best_score:
            best_score = score
            best_A = A
            
    # Convert best_A to sorted list
    best_set = [i+1 for i in range(n) if best_A[i]]
    
    # Final verification
    if count_4aps(best_A) != 0:
        # Fallback to greedy
        A = np.zeros(n, dtype=bool)
        candidates = list(range(n))
        random.shuffle(candidates)
        for x in candidates:
            valid = True
            for ap_idx in elem_to_aps4[x]:
                a, b, c, d = aps4[ap_idx]
                if a == x:
                    if A[b] and A[c] and A[d]:
                        valid = False; break
                elif b == x:
                    if A[a] and A[c] and A[d]:
                        valid = False; break
                elif c == x:
                    if A[a] and A[b] and A[d]:
                        valid = False; break
                else:
                    if A[a] and A[b] and A[c]:
                        valid = False; break
            if valid:
                A[x] = True
        best_set = [i+1 for i in range(n) if A[i]]

    return sorted(best_set)

# EVOLVE-BLOCK-END
