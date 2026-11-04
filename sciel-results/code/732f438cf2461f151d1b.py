import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = 100.0  # Safer margin
    
    # Base case
    if n == 1:
        return "1"
    
    word = [1]
    
    for k in range(2, n + 1):
        # Check if we are running out of time
        # If so, use a fast, non-optimal construction for the rest
        time_remaining = deadline - time.time()
        if time_remaining < 10.0:
            # Fast fallback: Use original order for this and all subsequent k
            # We need to finish the current k and then proceed to n
            # To be safe, we just use the fallback logic for the current k
            # and then continue the loop, but with a flag to skip TSP
            pass 

        if time.time() > deadline:
            # Absolute last resort: return a simple valid superpermutation if possible
            # But we are in the middle of the loop. 
            # We must ensure we return a valid S_n.
            # If we break here, word is S_{k-1}.
            # We cannot return S_{k-1} if k-1 < n.
            # So we should NOT break here if k < n.
            # Instead, we should force a fast completion.
            pass
            
        # Extract unique (k-1)-permutations from current word
        perms = []
        seen = set()
        for i in range(len(word) - k + 2):
            window = tuple(word[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        num_perms = len(perms)
        
        # Handle num_perms == 0 (should not happen, but safety)
        if num_perms == 0:
            # If no perms found, we can't build. Return current word (invalid for n) or dummy?
            # For safety, just append k to make it longer?
            # Actually, if num_perms is 0, word is too short.
            # This case shouldn't occur for n>=1.
            return "".join(map(str, word))

        # Determine if we should use fast mode
        time_remaining = deadline - time.time()
        use_fast_mode = (time_remaining < 5.0) or (num_perms > 100 and time_remaining < 30.0)
        
        if use_fast_mode:
            best_order = list(range(num_perms))
        else:
            # Greedy TSP
            best_order = None
            best_len = float('inf')
            
            # Try a few random starts
            if num_perms <= 20:
                starts = list(range(num_perms))
            else:
                starts = [random.randint(0, num_perms - 1) for _ in range(10)]
                
            for start_idx in starts:
                if time.time() > deadline:
                    break
                
            visited = [False] * num_perms
            order = []
            current = start_idx
            visited[current] = True
            order.append(current)
            
            for _ in range(num_perms - 1):
                if time.time() > deadline:
                    break
                best_next = -1
                best_overlap = -1
                p_curr = perms[current]
                
                for next_idx in range(num_perms):
                    if visited[next_idx]:
                        continue
                    p_next = perms[next_idx]
                    
                    # Calculate overlap
                    max_possible = min(len(p_curr), len(p_next))
                    overlap = 0
                    for o in range(min(k-2, max_possible), -1, -1):
                        if o == 0:
                            overlap = 0
                            break
                        if p_curr[-o:] == p_next[:o]:
                            overlap = o
                            break
                            
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_next = next_idx
                            
                if best_next == -1:
                    break
                visited[best_next] = True
                order.append(best_next)
                current = best_next
                
            if len(order) < num_perms:
                continue
                
            total_len = 0
            total_len += 2*(k-1) + 1
            
            for i in range(1, len(order)):
                p_prev = perms[order[i-1]]
                p_curr = perms[order[i]]
                max_possible = min(len(p_prev), len(p_curr))
                overlap = 0
                for o in range(min(k-2, max_possible), -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if p_prev[-o:] == p_curr[:o]:
                        overlap = o
                        break
                total_len += (2*(k-1) + 1) - overlap
            
            if total_len < best_len:
                best_len = total_len
                best_order = order
                
        if best_order is None:
            best_order = list(range(num_perms))
            
        # Construct new word
        new_word = []
        for idx, p_idx in enumerate(best_order):
            p = perms[p_idx]
            block = list(p) + [k] + list(p)
            
            if idx == 0:
                new_word = block
            else:
                max_overlap = min(len(new_word), len(block))
                overlap = 0
                for o in range(min(k, max_overlap), -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new_word[-o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
                
        word = new_word

    return "".join(map(str, word))
