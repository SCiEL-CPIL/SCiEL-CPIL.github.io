import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 256
    # present[i] is True if i is in the set
    present = [False] * (N + 1)
    S = []
    
    def can_add(x, current_S, present_arr):
        # Check if x can be added to current_S without forming a 3-AP
        for a in current_S:
            # Case 1: a, mid, x => mid = (a+x)/2
            if (a + x) % 2 == 0:
                mid = (a + x) // 2
                if 1 <= mid <= N and present_arr[mid]:
                    return False
            # Case 2: x, mid, a => mid = (x+a)/2 (Same as above)
            # Case 3: a, x, b => b = 2x - a
            b = 2 * x - a
            if 1 <= b <= N and b != x and present_arr[b]:
                return False
            # Case 4: x, a, b => b = 2a - x
            b2 = 2 * a - x
            if 1 <= b2 <= N and b2 != x and present_arr[b2]:
                return False
        return True

    # 1. Greedy Initialization
    for x in range(1, N + 1):
        if can_add(x, S, present):
            S.append(x)
            present[x] = True
            
    best_score = sum(1.0 / x for x in S)
    best_S = list(S)
    
    # 2. Local Search (1-flip)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_swap = None
        
        # Try all removals
        for i, a in enumerate(S):
            # Temporarily remove a
            present[a] = False
            S_temp = S[:i] + S[i+1:]
            
            # Try all additions
            for b in range(1, N + 1):
                if present[b]:
                    continue
                if can_add(b, S_temp, present):
                    gain = (1.0 / b) - (1.0 / a)
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_swap = (a, b)
            
            # Restore a
            present[a] = True
            S = S_temp + [a] if i < len(S_temp) else S_temp + [a] # Wait, S was modified? No, S_temp is a copy.
            # Actually, I modified S in place? No, S_temp is a new list.
            # But I need to restore S to its original state before next iteration.
            # My code above: S = S_temp + [a] is wrong logic if I don't manage indices carefully.
            # Let's just keep S as a list and use indices carefully or copy.
            # Since N is small, copying is fine.
            
        if best_swap:
            a, b = best_swap
            # Perform the swap
            S.remove(a)
            present[a] = False
            S.append(b)
            present[b] = True
            improved = True
            # Update best if this is better than global best (it should be, due to gain > 0)
            current_score = sum(1.0 / x for x in S)
            if current_score > best_score:
                best_score = current_score
                best_S = list(S)
        
        # Check time
        if time.time() > deadline:
            break

    return best_S

# Wait, the logic in the loop `S = S_temp + [a]` is messy.
# Let's rewrite the local search part cleanly.
