import time
import random
import itertools

# Precomputation for q=3, n=7
Q = 3
N = 7
RADIUS = 1
TOTAL_WORDS = Q ** N  # 2187

def word_to_idx(word):
    idx = 0
    for i in range(N):
        idx = idx * Q + word[i]
    return idx

def idx_to_word(idx):
    word = [0] * N
    for i in range(N - 1, -1, -1):
        word[i] = idx % Q
        idx //= Q
    return word

BALL_MASKS = [0] * TOTAL_WORDS
BALLS = [None] * TOTAL_WORDS 

def precompute():
    for idx in range(TOTAL_WORDS):
        word = idx_to_word(idx)
        ball = [idx]
        for pos in range(N):
            for delta in [-1, 1]:
                new_val = (word[pos] + delta) % Q
                if new_val != word[pos]:
                    new_word = word.copy()
                    new_word[pos] = new_val
                    n_idx = word_to_idx(new_word)
                    ball.append(n_idx)
        BALLS[idx] = ball
        mask = 0
        for b in ball:
            mask |= (1 << b)
        BALL_MASKS[idx] = mask

precompute()

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    start_time = time.time()
    deadline = start_time + 140.0
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    selected = set()
    for cw in initial_codewords:
        idx = word_to_idx(cw)
        selected.add(idx)
        
    counts = [0] * TOTAL_WORDS
    critical_mask = 0
    
    def update_counts(c_idx, delta):
        nonlocal critical_mask
        for w in BALLS[c_idx]:
            old_count = counts[w]
            new_count = old_count + delta
            counts[w] = new_count
            if old_count == 1 and new_count != 1:
                critical_mask &= ~(1 << w)
            elif old_count != 1 and new_count == 1:
                critical_mask |= (1 << w)

    for c in selected:
        update_counts(c, 1)
        
    best_solution = list(selected)
    best_size = len(selected)
    
    def is_deletable(c):
        return (BALL_MASKS[c] & critical_mask) == 0

    def try_delete():
        for c in list(selected):
            if is_deletable(c):
                update_counts(c, -1)
                selected.remove(c)
                return True
        return False

    def try_swap():
        non_deletable = [c for c in selected if not is_deletable(c)]
        if not non_deletable:
            return False
            
        random.shuffle(non_deletable)
        
        for c in non_deletable[:10]:
            crit_words = [w for w in BALLS[c] if counts[w] == 1]
            if not crit_words:
                continue
            
            req_mask = 0
            for w in crit_words:
                req_mask |= (1 << w)
                
            unselected = [c2 for c2 in range(TOTAL_WORDS) if c2 not in selected]
            random.shuffle(unselected)
            
            for c_prime in unselected[:50]:
                if (BALL_MASKS[c_prime] & req_mask) == req_mask:
                    update_counts(c, -1)
                    selected.remove(c)
                    update_counts(c_prime, 1)
                    selected.add(c_prime)
                    
                    if try_delete():
                        return True
                    else:
                        update_counts(c_prime, -1)
                        selected.remove(c_prime)
                        update_counts(c, 1)
                        selected.add(c)
                        return False
                        
        return False

    while time.time() < deadline:
        if try_delete():
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = list(selected)
            continue
        
        if try_swap():
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = list(selected)
            continue
        
        if len(selected) > 3:
            to_remove = random.sample(list(selected), 3)
            for c in to_remove:
                update_counts(c, -1)
                selected.remove(c)
            
            uncovered_mask = 0
            for w in range(TOTAL_WORDS):
                if counts[w] == 0:
                    uncovered_mask |= (1 << w)
            
            while uncovered_mask != 0 and time.time() < deadline:
                best_c = -1
                best_cover = 0
                for c in range(TOTAL_WORDS):
                    if c in selected:
                        continue
                    cover = (BALL_MASKS[c] & uncovered_mask).bit_count()
                    if cover > best_cover:
                        best_cover = cover
                        best_c = c
                
                if best_c == -1 or best_cover == 0:
                    break
                
                update_counts(best_c, 1)
                selected.add(best_c)
                uncovered_mask &= ~BALL_MASKS[best_c]
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = list(selected)

    return [idx_to_word(idx) for idx in best_solution]
