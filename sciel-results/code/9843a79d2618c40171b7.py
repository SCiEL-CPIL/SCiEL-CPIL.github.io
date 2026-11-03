import random
import math
import time

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        # Simple greedy
        total_words = q ** n
        neighbors = [[] for _ in range(total_words)]
        for w in range(total_words):
            nbrs = [w]
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            # digits[0] is pos 0 (LSB)
            for pos in range(n):
                div = q ** pos
                curr = digits[pos]
                for new_d in range(q):
                    if new_d != curr:
                        nbrs.append(w + (new_d - curr) * div)
            neighbors[w] = nbrs
            
        covered = [0] * total_words
        code = []
        while True:
            if all(c > 0 for c in covered):
                break
            best_w = -1
            best_gain = -1
            for w in range(total_words):
                if covered[w] > 0: continue # Optional optimization
                gain = sum(1 for nb in neighbors[w] if covered[nb] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_w = w
            if best_w == -1: break
            code.append(best_w)
            for nb in neighbors[best_w]:
                covered[nb] += 1
                
        final_code = []
        for w in code:
            word = [0] * n
            temp = w
            for i in range(n-1, -1, -1):
                word[i] = temp % q
                temp //= q
            final_code.append(word)
        return final_code

    # Optimized path for q=3, n=6, radius=1 using Simulated Annealing
    total_words = 729
    start_time = time.time()
    time_limit = 115 # 5s buffer
    
    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        nbrs = [w]
        for pos in range(n):
            div = 3 ** (n - 1 - pos)
            curr = (w // div) % 3
            for new_d in range(3):
                if new_d != curr:
                    nbrs.append(w + (new_d - curr) * div)
        neighbors[w] = nbrs

    def get_initial_greedy():
        covered = [0] * total_words
        code = []
        # Use a set for uncovered to speed up gain calculation?
        # No, just count.
        for _ in range(100): # Safety
            if all(c > 0 for c in covered):
                break
            best_w = -1
            best_gain = -1
            for w in range(total_words):
                gain = 0
                for nb in neighbors[w]:
                    if covered[nb] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_w = w
            if best_w == -1: break
            code.append(best_w)
            for nb in neighbors[best_w]:
                covered[nb] += 1
        return code

    # 1. Initialize with Greedy
    current_code = get_initial_greedy()
    
    # State for SA
    cover_counts = [0] * total_words
    for w in current_code:
        for nb in neighbors[w]:
            cover_counts[nb] += 1
            
    def calc_energy(code, cover_counts):
        uncovered = 0
        for c in cover_counts:
            if c == 0:
                uncovered += 1
        return len(code) * 1000 + uncovered

    current_energy = calc_energy(current_code, cover_counts)
    best_code = current_code[:]
    best_energy = current_energy
    
    # 2. Simulated Annealing
    T = 100.0
    cooling = 0.99
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit:
        if T < 0.5:
            break
            
        new_code = current_code[:]
        move_type = rng.random()
        
        if move_type < 0.4:
            # Remove
            if new_code:
                idx = rng.randrange(len(new_code))
                w = new_code.pop(idx)
                for nb in neighbors[w]:
                    cover_counts[nb] -= 1
        elif move_type < 0.7:
            # Add
            w = rng.randrange(total_words)
            # Check if already present (optimization: maintain a set? 
            # For small N, linear search is ok, but let's use a set for membership if we want speed)
            # Actually, just adding duplicates is harmless for coverage, but increases energy.
            # SA will reject high energy.
            new_code.append(w)
            for nb in neighbors[w]:
                cover_counts[nb] += 1
        else:
            # Swap
            if new_code:
                idx = rng.randrange(len(new_code))
                old_w = new_code[idx]
                new_w = rng.randrange(total_words)
                if new_w != old_w:
                    for nb in neighbors[old_w]:
                        cover_counts[nb] -= 1
                    for nb in neighbors[new_w]:
                        cover_counts[nb] += 1
                    new_code[idx] = new_w

        new_energy = calc_energy(new_code, cover_counts)
        delta = new_energy - current_energy
        
        if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            current_code = new_code
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_code = current_code[:]
                
        T *= cooling

    # Convert back to lists
    final_code = []
    for w in best_code:
        word = [0] * n
        temp = w
        for i in range(n-1, -1, -1):
            word[i] = temp % q
            temp //= q
        final_code.append(word)
        
    return final_code
