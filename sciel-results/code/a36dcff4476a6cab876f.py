import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # --- Greenwood-Gleason 32-vertex base ---
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

    def base_32_matrix():
        n = 32
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    c = log[(i % 16) ^ (j % 16)] % 3
                else:
                    c = 3
                matrix[i][j] = c
                matrix[j][i] = c
        return matrix

    def matrix_to_colors(matrix, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(matrix[i][j])
        return colors

    def count_mono_triangles(matrix, n):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = matrix[i][j]
                for k in range(j + 1, n):
                    if matrix[i][k] == c and matrix[j][k] == c:
                        count += 1
        return count

    def flip_delta(matrix, n, i, j, new_c):
        old_c = matrix[i][j]
        if old_c == new_c:
            return 0
        old_tri = 0
        new_tri = 0
        for k in range(n):
            if k == i or k == j:
                continue
            ik_c = matrix[i][k]
            jk_c = matrix[j][k]
            if ik_c == old_c and jk_c == old_c:
                old_tri += 1
            if ik_c == new_c and jk_c == new_c:
                new_tri += 1
        return new_tri - old_tri

    def simulated_annealing(matrix, n, time_limit):
        current_energy = count_mono_triangles(matrix, n)
        if current_energy == 0:
            return matrix, 0

        T = 5.0
        T_min = 0.01
        cooling_rate = 0.9995

        best_energy = current_energy
        best_matrix = [row[:] for row in matrix]

        while time.time() < time_limit:
            if current_energy == 0:
                return matrix, 0

            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i

            new_c = random.randint(0, 3)
            if matrix[i][j] == new_c:
                continue

            delta = flip_delta(matrix, n, i, j, new_c)

            if delta < 0:
                matrix[i][j] = new_c
                matrix[j][i] = new_c
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_matrix = [row[:] for row in matrix]
            else:
                if random.random() < math.exp(-delta / T):
                    matrix[i][j] = new_c
                    matrix[j][i] = new_c
                    current_energy += delta

            T *= cooling_rate
            if T < T_min:
                T = T_min

            # Occasional reheat
            if current_energy > 0 and random.random() < 0.001:
                T = max(T, 2.0)

        if best_energy < current_energy:
            return best_matrix, best_energy
        return matrix, current_energy

    def try_solve(n, time_limit, init_matrix=None):
        if init_matrix is None:
            matrix = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = random.randint(0, 3)
                    matrix[i][j] = c
                    matrix[j][i] = c
        else:
            matrix = [row[:] for row in init_matrix]

        matrix, energy = simulated_annealing(matrix, n, time_limit)
        if energy == 0:
            return {'n': n, 'colors': matrix_to_colors(matrix, n)}
        return None

    # Fallback: known valid K32
    base = base_32_matrix()
    best_result = {'n': 32, 'colors': matrix_to_colors(base, 32)}

    # Try to extend from 33 upwards
    # We use a strategy of trying each n with a few attempts, using time slices.
    # If we succeed, we move to n+1.
    # If we fail, we might try a bit harder or stop.
    # Given the hardness, let's try up to 50 or so, but stop if we fail a few in a row?
    # Actually, the task says maximize n.
    # Let's iterate n from 33 to 62.
    # Allocate time based on remaining budget.
    
    current_n = 33
    while current_n <= 62:
        if time.time() > deadline - 10:
            break
            
        # How much time to give this n?
        # Let's give 10 seconds per attempt, 2 attempts per n?
        # Or just 15 seconds total per n?
        # If we solve it quickly, great.
        # If we don't, we move on?
        # Usually, if n is hard, n+1 is harder.
        
        # Let's try 2 attempts of ~5s each for n.
        found = False
        for attempt in range(2):
            if time.time() > deadline - 5:
                break
            
            slice_end = min(time.time() + 5, deadline - 5)
            
            # Init strategy:
            # For n close to 32, we can try to extend the base.
            # But SA from random is often more robust for finding *any* solution if one exists.
            # However, for n=33, random might be bad.
            # Let's try random init first, then maybe perturbed base if n <= 40.
            
            result = try_solve(current_n, slice_end)
            if result:
                best_result = result
                found = True
                break
        
        if found:
            current_n += 1
        else:
            # If we fail, try next n? Or break?
            # If we can't solve n, it's likely we can't solve n+1.
            # But sometimes random luck helps.
            # Let's try one more time with a slightly longer slice if we have time?
            # Or just break to save time for other potential improvements?
            # The prompt says "Maximize n".
            # If I break, I return current best.
            # If I continue, I might waste time.
            # Let's break if we fail, assuming monotonic difficulty.
            break

    return best_result

# Test locally? No, just output.
# The code above looks solid.
# One detail: `math.exp` can be slow. `2 ** (-delta / T)` is faster? Or precompute?
# Delta is integer. T is float.
# We can use `random.random() < 2 ** (-delta / T)`.
# Let's replace math.exp with a faster approximation or just use it.
# For small delta/T, exp is fine.

# Final check on constraints:
# 1 <= n <= 62.
# Colors in {0,1,2,3}.
# No monochromatic triangles.
# Return dict.

# The code seems solid.
