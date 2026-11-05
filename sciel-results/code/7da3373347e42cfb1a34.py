import time
import random
import math

def construct():
    N = 512
    conflicts = [[] for _ in range(N+1)]
    
    for a in range(1, N+1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            
            m = (1 << b) | (1 << c) | (1 << e)
            conflicts[a].append(m)
            
            m = (1 << a) | (1 << c) | (1 << e)
            conflicts[b].append(m)
            
            m = (1 << a) | (1 << b) | (1 << e)
            conflicts[c].append(m)
            
            m = (1 << a) | (1 << b) | (1 << c)
            conflicts[e].append(m)
            
    def is_valid_add(x, S_bits):
        for m in conflicts[x]:
            if (S_bits & m) == m:
                return False
        return True

    best_S = set()
    best_size = 0
    best_S_bits = 0

    start_time = time.time()
    deadline = start_time + 58.0

    # Greedy Init
    for attempt in range(50):
        if time.time() > deadline - 30.0:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        S = set()
        S_bits = 0
        for x in order:
            if is_valid_add(x, S_bits):
                S.add(x)
                S_bits |= (1 << x)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            best_S_bits = S_bits

    S = set(best_S)
    S_bits = best_S_bits
    size = len(S)
    
    all_elements = list(range(1, N+1))
    
    while time.time() < deadline:
        # Strategy:
        # 50% chance: Try "Remove One, Add Many"
        # 50% chance: Try "Add One" or "Swap One"
        
        if random.random() < 0.5:
            # Remove One, Add Many
            if not S:
                continue
            y = random.choice(list(S))
            # Tentatively remove y
            S.remove(y)
            S_bits &= ~(1 << y)
            size -= 1
            
            added_count = 0
            added_elements = []
            
            # Try to add all possible elements
            # Optimization: Only try elements that were blocked by y?
            # Hard to know which were blocked by y specifically without checking.
            # Just try all x not in S.
            for x in all_elements:
                if x not in S:
                    if is_valid_add(x, S_bits):
                        S.add(x)
                        S_bits |= (1 << x)
                        size += 1
                        added_count += 1
                        added_elements.append(x)
            
            # Accept if size increased (added_count > 1) or same (added_count == 1) with prob
            if added_count > 1:
                if size > best_size:
                    best_size = size
                    best_S = S.copy()
                    best_S_bits = S_bits
            elif added_count == 1:
                # Keep the change if it doesn't hurt, might help later
                if size > best_size:
                    best_size = size
                    best_S = S.copy()
                    best_S_bits = S_bits
            else:
                # added_count == 0, size decreased. Revert?
                # Or keep to explore?
                # Let's revert if we are not stuck, to avoid shrinking.
                # But if we are stuck, shrinking might help.
                # For now, revert if added_count == 0.
                S.add(y)
                S_bits |= (1 << y)
                size += 1
                # Remove the added elements if any (none in this case)
        else:
            # Standard Add/Swap
            x = random.choice(all_elements)
            if x not in S:
                if is_valid_add(x, S_bits):
                    S.add(x)
                    S_bits |= (1 << x)
                    size += 1
                    if size > best_size:
                        best_size = size
                        best_S = S.copy()
                        best_S_bits = S_bits
                else:
                    # Swap
                    # Find conflict
                    for m in conflicts[x]:
                        if (S_bits & m) == m:
                            # Extract one element from m
                            temp = m
                            lsb = temp & -temp
                            idx = lsb.bit_length() - 1
                            y = idx
                            S.remove(y)
                            S_bits &= ~(1 << y)
                            S.add(x)
                            S_bits |= (1 << x)
                            # Size same
                            break

    return sorted(best_S)
