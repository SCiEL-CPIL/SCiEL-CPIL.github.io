import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 120s limit
    
    # Fallback: Seed program's N=32 solution
    def get_seed_solution():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}

    # Target N=50
    N = 50
    # colors is a 2D list NxN, colors[i][j] for i != j
    # We only need i < j, but symmetric storage is easier for logic
    colors = [[0] * N for _ in range(N)]
    
    def init_random():
        for i in range(N):
            for j in range(i + 1, N):
                c = random.randint(0, 3)
                colors[i][j] = c
                colors[j][i] = c

    def count_mono_triangles():
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                c1 = colors[i][j]
                for k in range(j + 1, N):
                    if colors[i][k] == c1 and colors[j][k] == c1:
                        count += 1
        return count

    def calculate_delta(i, j, new_c):
        """
        Calculate change in number of mono triangles if we set colors[i][j] = new_c.
        Current color is colors[i][j].
        """
        old_c = colors[i][j]
        if old_c == new_c:
            return 0
        
        delta = 0
        for k in range(N):
            if k == i or k == j:
                continue
            c_ik = colors[i][k]
            c_jk = colors[j][k]
            
            # Check if triangle (i, j, k) was mono with old_c
            was_mono = (c_ik == old_c and c_jk == old_c)
            # Check if triangle (i, j, k) will be mono with new_c
            is_mono = (c_ik == new_c and c_jk == new_c)
            
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
        
        return delta

    # Local Search
    init_random()
    current_cost = count_mono_triangles()
    
    best_solution = None
    best_cost = current_cost
    
    # If we start with 0 (unlikely), return immediately
    if current_cost == 0:
        # Flatten and return
        flat_colors = []
        for i in range(N):
            for j in range(i + 1, N):
                flat_colors.append(colors[i][j])
        return {'n': N, 'colors': flat_colors}

    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Pick a random edge
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Try all 4 colors
        best_new_c = colors[i][j]
        best_delta = 0
        for c in range(4):
            delta = calculate_delta(i, j, c)
            if delta < best_delta:
                best_delta = delta
                best_new_c = c
        
        # If we found an improvement
        if best_delta < 0:
            colors[i][j] = best_new_c
            colors[j][i] = best_new_c
            current_cost += best_delta
            no_improve_count = 0
            
            if current_cost == 0:
                flat_colors = []
                for i in range(N):
                    for j in range(i + 1, N):
                        flat_colors.append(colors[i][j])
                return {'n': N, 'colors': flat_colors}
            
            if current_cost < best_cost:
                best_cost = current_cost
                # Save best solution found so far? 
                # We only return if cost 0, so we don't need to save intermediate states 
                # unless we want to return a partial solution if we run out of time.
                # But partial solutions are invalid (score 0). 
                # So we only care about cost 0.
        
        else:
            no_improve_count += 1
            # If stuck, do a random perturbation
            if no_improve_count > max_no_improve:
                # Restart with new random init
                init_random()
                current_cost = count_mono_triangles()
                no_improve_count = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                
                if current_cost == 0:
                    flat_colors = []
                    for i in range(N):
                        for j in range(i + 1, N):
                            flat_colors.append(colors[i][j])
                    return {'n': N, 'colors': flat_colors}

    # If we didn't find a solution for N=50, return the seed solution (N=32)
    # This ensures a valid result and score 1.0
    return get_seed_solution()

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(res['n'])
