import time
import random

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 55.0
    
    def check_valid_addition(x, S_list, S_mask):
        # Returns True if x can be added to S_list without violating constraints
        # x is not in S_list
        temp_mask = S_mask | (1 << x)
        
        # We need to check if there exist distinct u, v in S_list
        # such that u+v, x+u, x+v are in temp_mask
        # Note: u, v must be distinct from each other and x.
        # Since x is not in S_list, u, v != x is automatic.
        
        for i in range(len(S_list)):
            u = S_list[i]
            xu = x + u
            if xu > N:
                continue
            if not ((temp_mask >> xu) & 1):
                continue
            
            for j in range(i + 1, len(S_list)):
                v = S_list[j]
                xv = x + v
                if xv > N:
                    continue
                if not ((temp_mask >> xv) & 1):
                    continue
                
                uv = u + v
                if uv > N:
                    continue
                if (temp_mask >> uv) & 1:
                    return False
        return True

    best_size = 0
    best_set = []
    
    # Initial greedy with sorted order to get a baseline
    for order_type in [0, 1]:
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        if order_type == 1:
            order.reverse()
        else:
            random.shuffle(order)
            
        S_list = []
        S_mask = 0
        for x in order:
            if check_valid_addition(x, S_list, S_mask):
                S_list.append(x)
                S_mask |= (1 << x)
        
        # Local Search
        while time.time() < deadline:
            added = False
            # Try to add a random valid element
            for _ in range(50):
                x = random.randint(1, N)
                if (S_mask >> x) & 1:
                    continue
                if check_valid_addition(x, S_list, S_mask):
                    S_list.append(x)
                    S_mask |= (1 << x)
                    added = True
                    if len(S_list) > best_size:
                        best_size = len(S_list)
                        best_set = list(S_list)
                    break
            
            if not added:
                # Try a kick: remove one, add one
                if S_list:
                    y = random.choice(S_list)
                    S_list.remove(y)
                    S_mask &= ~(1 << y)
                    
                    kick_success = False
                    for _ in range(50):
                        x = random.randint(1, N)
                        if x == y:
                            continue
                        if (S_mask >> x) & 1:
                            continue
                        if check_valid_addition(x, S_list, S_mask):
                            S_list.append(x)
                            S_mask |= (1 << x)
                            kick_success = True
                            break
                    
                    if not kick_success:
                        # Revert
                        S_list.append(y)
                        S_mask |= (1 << y)
                else:
                    break

    return best_set
