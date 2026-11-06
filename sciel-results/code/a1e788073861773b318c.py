"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute gcd matrix for all pairs in [1, 128]
    n = 128
    gcd_matrix = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
    
    # Check if a set of numbers is valid (no three distinct elements with all three pairwise gcds equal)
    def is_valid_set(s):
        s_list = list(s)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                for k in range(j + 1, len(s_list)):
                    a, b, c = s_list[i], s_list[j], s_list[k]
                    if gcd_matrix[a][b] == gcd_matrix[a][c] == gcd_matrix[b][c]:
                        return False
        return True
    
    # Greedy construction with multiple orderings
    def greedy(order):
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            # Check if adding x creates a bad triple
            valid = True
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    a, b = selected[i], selected[j]
                    if gcd_matrix[a][b] == gcd_matrix[a][x] == gcd_matrix[b][x]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                selected.append(x)
        return selected
    
    # Try different orderings
    orderings = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        [x for x in range(1, 129) if x % 2 == 1],
        [x for x in range(1, 129) if x % 2 == 0],
        sorted(range(1, 129), key=lambda x: -x.bit_count()),
        sorted(range(1, 129), key=lambda x: x.bit_count()),
        [x for x in range(1, 129) if x % 3 != 0],
        [x for x in range(1, 129) if x % 3 == 0],
        [x for x in range(1, 129) if x % 4 != 0],
        [x for x in range(1, 129) if x % 4 == 0],
    ]
    
    best = set()
    for order in orderings:
        if time.time() > deadline:
            break
        result = greedy(order)
        if len(result) > len(best):
            best = set(result)
    
    # Local search: try to add more elements
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current = list(best)
            for x in range(1, 129):
                if time.time() > deadline - 5:
                    break
                if x in best:
                    continue
                # Try adding x
                new_set = best | {x}
                if is_valid_set(new_set):
                    best = new_set
                    improved = True
                    break
    
    # Try to swap elements to improve
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            improved = False
            current = list(best)
            for x in current:
                if time.time() > deadline - 10:
                    break
                for y in range(1, 129):
                    if time.time() > deadline - 10:
                        break
                    if y == x or y in best:
                        continue
                    new_set = (best - {x}) | {y}
                    if is_valid_set(new_set) and len(new_set) == len(best):
                        # Check if this swap enables adding more
                        can_add = False
                        for z in range(1, 129):
                            if z in new_set:
                                continue
                            if is_valid_set(new_set | {z}):
                                can_add = True
                                break
                        if can_add:
                            best = new_set
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
    
    # Final local search: try to add more elements
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in range(1, 129):
                if time.time() > deadline - 5:
                    break
                if x in best:
                    continue
                new_set = best | {x}
                if is_valid_set(new_set):
                    best = new_set
                    improved = True
                    break
    
    return list(best) if best else [1]
# EVOLVE-BLOCK-END
