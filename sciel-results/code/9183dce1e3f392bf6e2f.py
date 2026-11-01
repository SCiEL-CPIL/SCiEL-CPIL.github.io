import random
import math

def construct(n_rows, k):
    """
    Build n_rows rows of k+1 increasing integers starting at 0,
    such that all positive within-row differences are globally distinct.
    Minimize the largest entry.
    """
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def generate_initial():
        used_diffs = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows

    initial_rows = generate_initial()
    best_rows = [row[:] for row in initial_rows]
    best_cost = max(max(row) for row in best_rows)
    
    current_rows = [row[:] for row in initial_rows]
    current_cost = best_cost
    
    current_row_diffs = [get_diffs(row) for row in current_rows]
    current_global_diffs = set()
    for rd in current_row_diffs:
        current_global_diffs |= rd
        
    row_maxes = [max(row) for row in current_rows]
    
    start_temp = 50.0
    end_temp = 0.1
    iterations = 200000 # Increase iterations, they are fast now
    
    for it in range(iterations):
        temp = start_temp * (end_temp / start_temp) ** (it / iterations)
        
        r = random.randint(0, n_rows - 1)
        i = random.randint(1, k)
        
        delta_range = 20
        delta = random.randint(-delta_range, delta_range)
        new_val = current_rows[r][i] + delta
        
        if new_val <= current_rows[r][i-1] or (i < k and new_val >= current_rows[r][i+1]):
            continue
        if new_val <= 0:
            continue

        new_row = current_rows[r][:]
        new_row[i] = new_val
        
        new_row_diffs = get_diffs(new_row)
        
        other_diffs = current_global_diffs - current_row_diffs[r]
        
        if new_row_diffs & other_diffs:
            continue
            
        new_row_max = max(new_row)
        # Calculate new global max
        # max of other rows + new_row_max
        other_max = 0
        for j in range(n_rows):
            if j != r:
                if row_maxes[j] > other_max:
                    other_max = row_maxes[j]
        new_cost = max(new_row_max, other_max)
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / temp):
            # Accept
            current_rows[r] = new_row
            current_cost = new_cost
            row_maxes[r] = new_row_max
            
            current_row_diffs[r] = new_row_diffs
            current_global_diffs = set()
            for rd in current_row_diffs:
                current_global_diffs |= rd
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [row[:] for row in current_rows]

    return best_rows
