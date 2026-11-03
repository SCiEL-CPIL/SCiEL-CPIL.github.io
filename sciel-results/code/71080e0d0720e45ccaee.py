import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Theoretical lower bound for the maximum entry
    # Total differences = n_rows * C(k+1, 2)
    # These must be distinct integers >= 1.
    # So the largest difference must be at least total_diffs.
    # The largest entry in a row is at least the largest difference in that row.
    # So max_entry >= total_diffs.
    total_diffs = n_rows * (k + 1) * k // 2
    lower_bound = total_diffs

    def get_cost(rows):
        diffs = set()
        max_val = 0
        for r in rows:
            if r:
                max_val = max(max_val, max(r))
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.add(r[j] - r[i])
        collisions = total_diffs - len(diffs)
        return max_val + 10000 * collisions

    def greedy_solution():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows

    current_rows = greedy_solution()
    best_rows = [r[:] for r in current_rows]
    best_cost = get_cost(best_rows)
    current_cost = best_cost

    if best_cost == lower_bound:
        return best_rows

    T = 100.0
    cooling = 0.99
    min_T = 0.5

    # To escape local minima, we can restart temperature if stuck
    last_improve_time = start_time

    while time.time() < deadline:
        if time.time() - last_improve_time > 10.0:
            # Restart temperature
            T = 50.0
            last_improve_time = time.time()

        i = random.randint(0, n_rows - 1)
        j = random.randint(1, k)
        
        old_val = current_rows[i][j]
        
        lower = current_rows[i][j-1] + 1
        upper = current_rows[i][j+1] - 1 if j < k else old_val + 50
        
        if lower > upper:
            continue
            
        # Random step
        step = random.randint(-20, 20)
        new_val = old_val + step
        
        if new_val < lower or new_val > upper:
            continue
            
        current_rows[i][j] = new_val
        new_cost = get_cost(current_rows)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [r[:] for r in current_rows]
                last_improve_time = time.time()
                if best_cost == lower_bound:
                    break
        else:
            delta = new_cost - current_cost
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_cost = new_cost
                else:
                    current_rows[i][j] = old_val
        
        T = max(min_T, T * cooling)

    return best_rows
