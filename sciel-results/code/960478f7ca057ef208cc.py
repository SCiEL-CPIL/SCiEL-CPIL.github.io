import itertools
import time

def construct(n):
    if n <= 4:
        # Standard construction is optimal for n <= 4
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n > 4, use TSP heuristic on the last step
    # 1. Construct superpermutation for n-1
    word_prev = construct(n - 1)
    word_list = [int(c) for c in word_prev]
    
    # 2. Extract permutations of 1..n-1 in order of appearance
    m = n - 1
    perms, seen = [], set()
    for i in range(len(word_list) - m + 1):
        window = tuple(word_list[i:i + m])
        if len(set(window)) == m and window not in seen:
            seen.add(window)
            perms.append(window)
            
    # 3. Compute overlap matrix
    # W[i][j] = overlap between perms[i] and perms[j]
    # Overlap is longest suffix of perms[i] that is prefix of perms[j]
    N = len(perms)
    W = [[0] * N for _ in range(N)]
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            pj = perms[j]
            # Max overlap is m-1
            max_ov = m - 1
            for k in range(max_ov, -1, -1):
                if k == 0:
                    W[i][j] = 0
                    break
                if pi[m-k:] == pj[:k]:
                    W[i][j] = k
                    break

    # 4. Find order maximizing sum of overlaps using 2-opt
    # Initial order: 0, 1, ..., N-1 (standard order)
    order = list(range(N))
    
    def calc_sum(order):
        s = 0
        for i in range(N - 1):
            s += W[order[i]][order[i+1]]
        return s

    current_sum = calc_sum(order)
    
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s for construction and safety
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, N):
                # Try reversing order[i+1..j]
                # Change in sum:
                # Remove edges (i, i+1) and (j, j+1)
                # Add edges (i, j) and (i+1, j+1)
                # Note: if j == N-1, there is no (j, j+1) edge
                old_sum = 0
                new_sum = 0
                
                # Edge (i, i+1)
                if i > 0:
                    old_sum += W[order[i-1]][order[i]]
                    new_sum += W[order[i-1]][order[j]] # (i-1) connects to j? No.
                    
                # Let's just recalculate the local change carefully.
                # Path: ... A - B ... C - D ...
                # A=order[i], B=order[i+1], C=order[j], D=order[j+1]
                # Edges removed: (A,B) and (C,D) [if D exists]
                # Edges added: (A,C) and (B,D) [if D exists]
                
                # Actually, reversing B...C means:
                # Original: ... A, B, ..., C, D ...
                # New: ... A, C, ..., B, D ...
                
                # Change in sum:
                delta = W[order[i]][order[j]] + W[order[i+1]][order[j+1]] - W[order[i]][order[i+1]] - W[order[j]][order[j+1]]
                
                # Handle boundaries
                if i == 0:
                    # No edge before A? No, A is start.
                    # Edges: (A,B) removed. (A,C) added.
                    # Wait, if i=0, A is start.
                    # Original: A-B...
                    # New: A-C...
                    # Removed: (A,B). Added: (A,C).
                    # And at the other end:
                    # Original: ...C-D...
                    # New: ...B-D...
                    # Removed: (C,D). Added: (B,D).
                    # So delta = W[A][C] + W[B][D] - W[A][B] - W[C][D]
                    # This formula holds if D exists.
                    pass # The formula above works for i=0 if we treat "edge before A" as 0, but we don't add it.
                    # The formula delta = W[order[i]][order[j]] + W[order[i+1]][order[j+1]] - W[order[i]][order[i+1]] - W[order[j]][order[j+1]]
                    # assumes edges (i, i+1) and (j, j+1) exist.
                    # If i=0, edge (i-1, i) doesn't exist, but we don't use it.
                    # The edges involved are (i, i+1) and (j, j+1).
                    # So the formula is correct for all i < j.
                    
                if j == N - 1:
                    # D doesn't exist.
                    # Original: ...A-B...C (end)
                    # New: ...A-C...B (end)
                    # Removed: (A,B) and (C, nothing=0).
                    # Added: (A,C) and (B, nothing=0).
                    # So delta = W[A][C] - W[A][B].
                    # Formula: W[A][C] + W[B][0] - W[A][B] - W[C][0]
                    # If we define W[x][N] = 0, it works.
                    pass

                # To be safe and simple, just calculate delta explicitly for boundaries
                if i == 0:
                    # Start
                    delta = W[order[i]][order[j]] - W[order[i]][order[i+1]]
                    if j < N - 1:
                        delta += W[order[i+1]][order[j+1]] - W[order[j]][order[j+1]]
                    else:
                        # j is end
                        delta += 0 # W[B][end] is 0
                elif j == N - 1:
                    # End
                    delta = W[order[i+1]][order[j+1]] - W[order[j]][order[j+1]] if j < N-1 else 0
                    # Wait, if j == N-1, D is out of bounds.
                    # Original: ...A-B...C (end)
                    # New: ...A-C...B (end)
                    # Removed: (A,B) and (C, 0).
                    # Added: (A,C) and (B, 0).
                    # Delta = W[A][C] - W[A][B]
                    delta = W[order[i]][order[j]] - W[order[i]][order[i+1]]
                else:
                    # Middle
                    delta = W[order[i]][order[j]] + W[order[i+1]][order[j+1]] - W[order[i]][order[i+1]] - W[order[j]][order[j+1]]

                if delta > 0:
                    order[i+1:j+1] = reversed(order[i+1:j+1])
                    current_sum += delta
                    improved = True
                    if time.time() > deadline:
                        break

    # 5. Construct final word
    word = list(word_list)
    for idx in order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        overlap = 0
        max_ov = min(len(word), len(block))
        for o in range(max_ov, -1, -1):
            if word[len(word)-o:] == block[:o]:
                overlap = o
                break
        word += block[overlap:]
        
    return "".join(map(str, word))
