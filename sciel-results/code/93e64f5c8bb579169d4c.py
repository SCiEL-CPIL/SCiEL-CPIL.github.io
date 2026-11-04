"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # First, get a baseline solution using the greedy approach
    baseline = _greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline) if baseline else float('inf')
    
    # Try to improve with randomized search
    best_solution = baseline
    best_scope = baseline_scope
    
    # For small problem sizes, try local search / simulated annealing
    if n_rows <= 10 and k <= 8:
        improved = _randomized_search(n_rows, k, best_scope, deadline)
        if improved is not None:
            scope = max(max(row) for row in improved)
            if scope < best_scope:
                best_solution = improved
                best_scope = scope
    
    return best_solution

def _greedy_construct(n_rows, k):
    """Original greedy approach."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _randomized_search(n_rows, k, initial_scope, deadline):
    """Randomized local search to improve upon the greedy solution."""
    random.seed(42)
    
    # Start from the greedy solution
    current = _greedy_construct(n_rows, k)
    current_scope = max(max(row) for row in current)
    
    best = current[:]
    best_scope = current_scope
    
    # Define a neighborhood move: swap two elements in a row, or adjust one element
    def get_neighbors(solution):
        neighbors = []
        for i, row in enumerate(solution):
            # Try moving one element
            for j in range(1, len(row)):
                for delta in [-2, -1, 1, 2, 3]:
                    new_val = row[j] + delta
                    if new_val > row[j-1] and (j == len(row)-1 or new_val < row[j+1]):
                        new_row = row[:]
                        new_row[j] = new_val
                        new_solution = solution[:]
                        new_solution[i] = new_row
                        if _is_valid_solution(new_solution):
                            neighbors.append(new_solution)
        return neighbors
    
    def _is_valid_solution(solution):
        """Check if all differences are distinct across all rows."""
        used_diffs = set()
        for row in solution:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff = row[j] - row[i]
                    if diff in used_diffs:
                        return False
                    used_diffs.add(diff)
        return True
    
    # Simulated annealing
    temperature = 50.0
    cooling_rate = 0.995
    min_temp = 0.1
    
    while time.time() < deadline and temperature > min_temp:
        # Generate a random neighbor
        i = random.randint(0, len(current) - 1)
        j = random.randint(1, len(current[i]) - 1)
        delta = random.choice([-3, -2, -1, 1, 2, 3])
        
        new_val = current[i][j] + delta
        if new_val > current[i][j-1] and (j == len(current[i])-1 or new_val < current[i][j+1]):
            new_solution = [row[:] for row in current]
            new_solution[i][j] = new_val
            
            if _is_valid_solution(new_solution):
                new_scope = max(max(row) for row in new_solution)
                delta_cost = new_scope - current_scope
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / temperature):
                    current = new_solution
                    current_scope = new_scope
                    
                    if current_scope < best_scope:
                        best = [row[:] for row in current]
                        best_scope = current_scope
        
        temperature *= cooling_rate
    
    return best if best_scope < initial_scope else None
# EVOLVE-BLOCK-END
