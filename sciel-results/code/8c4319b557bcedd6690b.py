import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = 115.0
    
    total_words = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    balls = []
    ball_sets = []
    for w in words:
        ball = [word_to_idx[w]]
        for i in range(n):
            for val in range(q):
                if val == w[i]: continue
                new_w_list = list(w)
                new_w_list[i] = val
                new_w = tuple(new_w_list)
                ball.append(word_to_idx[new_w])
        balls.append(ball)
        ball_sets.append(set(ball))
        
    cover_count = [0] * total_words
    code = []
    uncovered_count = total_words
    
    # 1. Greedy
    while uncovered_count > 0 and time.time() < deadline:
        best_c = -1
        best_gain = -1
        for c in range(total_words):
            gain = 0
            for u in balls[c]:
                if cover_count[u] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
            if gain == 0:
                # Optimization: if we find a 0, it's not the best if others are > 0.
                # But we don't know others.
                pass
        
        if best_gain == 0:
            break
            
        code.append(best_c)
        for u in balls[best_c]:
            if cover_count[u] == 0:
                uncovered_count -= 1
            cover_count[u] += 1

    # 2. Removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            redundant = True
            for u in balls[c]:
                if cover_count[u] == 1:
                    redundant = False
                    break
            
            if redundant:
                for u in balls[c]:
                    cover_count[u] -= 1
                code.pop(i)
                changed = True

    # 3. Swap (2 -> 1 or 2 -> 0)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        L = len(code)
        for i in range(L):
            if time.time() > deadline: break
            for j in range(i + 1, L):
                if time.time() > deadline: break
                
                c1 = code[i]
                c2 = code[j]
                
                # Decrement counts
                for u in balls[c1]:
                    cover_count[u] -= 1
                for u in balls[c2]:
                    cover_count[u] -= 1
                
                # Find critical words
                critical_words = set()
                for u in balls[c1]:
                    if cover_count[u] == 0:
                        critical_words.add(u)
                for u in balls[c2]:
                    if cover_count[u] == 0:
                        critical_words.add(u)
                        
                move_made = False
                if not critical_words:
                    # Can remove both
                    code.pop(j)
                    code.pop(i)
                    improved = True
                    move_made = True
                else:
                    # Need to find k covering critical_words
                    for k in range(total_words):
                        if critical_words.issubset(ball_sets[k]):
                            # Found k
                            # Add k
                            code.append(k)
                            for u in balls[k]:
                                cover_count[u] += 1
                            # Remove c1, c2
                            code.pop(j)
                            code.pop(i)
                            improved = True
                            move_made = True
                            break
                            
                if move_made:
                    # Restart swap loop
                    break
                else:
                    # Restore counts
                    for u in balls[c1]:
                        cover_count[u] += 1
                    for u in balls[c2]:
                        cover_count[u] += 1
                        
            if improved:
                break
                
    return [list(words[c]) for c in code]
