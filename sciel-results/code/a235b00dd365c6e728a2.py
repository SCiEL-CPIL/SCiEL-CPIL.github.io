import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 58.0
    
    # Precompute all 4-term arithmetic progressions
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(all_aps)
    
    # Map each number to the indices of APs it belongs to
    ap_indices_by_num = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(all_aps):
        for x in ap:
            ap_indices_by_num[x].append(idx)
    
    # State for fast add/remove
    ap_count = [0] * num_aps
    in_set = [False] * (N + 1)
    active_elements = []
    current_size = 0
    
    def add_element(x):
        nonlocal current_size
        if in_set[x]:
            return True
        # Check if adding x creates a 4-AP
        for ap_idx in ap_indices_by_num[x]:
            if ap_count[ap_idx] == 3:
                return False
        # Add x
        in_set[x] = True
        current_size += 1
        active_elements.append(x)
        for ap_idx in ap_indices_by_num[x]:
            ap_count[ap_idx] += 1
        return True

    def remove_element(x):
        nonlocal current_size
        if not in_set[x]:
            return True
        in_set[x] = False
        current_size -= 1
        # Remove from active_elements
        # O(N) operation, acceptable for N=512
        try:
            active_elements.remove(x)
        except ValueError:
            pass
        for ap_idx in ap_indices_by_num[x]:
            ap_count[ap_idx] -= 1
        return True

    def reset_state():
        nonlocal ap_count, in_set, active_elements, current_size
        ap_count = [0] * num_aps
        in_set = [False] * (N + 1)
        active_elements = []
        current_size = 0

    best_size = 0
    best_set = []

    # Phase 1: Multiple Randomized Greedy Starts
    # We can do many restarts because each is fast
    num_restarts = 2000
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0: # Leave 5s for final check/safety
            break
            
        reset_state()
        
        # Random order
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            add_element(x)
        
        # Phase 2: Local Search (Polishing)
        # Try to add more elements or swap to improve
        # Simple local search: try to add random elements, if fail, remove random and retry
        for _ in range(100): # 100 steps of local search per restart
            if time.time() > deadline - 5.0:
                break
            
            # Try to add a random element not in set
            # Pick random x from 1..N
            x = random.randint(1, N)
            
            if not in_set[x]:
                if add_element(x):
                    if current_size > best_size:
                        best_size = current_size
                        best_set = active_elements.copy()
                else:
                    # If adding failed, try to remove a random element from active set
                    # and then try to add x again
                    if active_elements:
                        u = random.choice(active_elements)
                        remove_element(u)
                        # Try to add x again
                        if add_element(x):
                            if current_size > best_size:
                                best_size = current_size
                                best_set = active_elements.copy()
                        else:
                            # If still failed, we've just swapped u for nothing (size decreased)
                            # Revert: add u back? Or keep size decreased to explore?
                            # Keeping size decreased might allow future additions.
                            # But we want to maximize. 
                            # Let's revert to keep size constant if possible?
                            # Actually, if we removed u and couldn't add x, we are worse off.
                            # Revert.
                            add_element(u)

    return sorted(best_set)
