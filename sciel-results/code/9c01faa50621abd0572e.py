import time
from fractions import Fraction
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Weights
    weights = {i: Fraction(1, i) for i in range(1, N + 1)}
    
    def can_add(x, S, in_set):
        """Check if x can be added to S without creating a 3-AP."""
        for y in S:
            # Case 1: y, x, z => z = 2x - y
            z = 2 * x - y
            if 1 <= z <= N and in_set[z]:
                return False
            # Case 2: x, y, z => z = 2y - x
            z = 2 * y - x
            if 1 <= z <= N and in_set[z]:
                return False
            # Case 3: y, z, x => z = (x+y)/2
            s = x + y
            if s % 2 == 0:
                z = s // 2
                if 1 <= z <= N and in_set[z]:
                    return False
        return True

    # Greedy Initialization
    S = set()
    in_set = [False] * (N + 1)
    
    for x in range(1, N + 1):
        if can_add(x, S, in_set):
            S.add(x)
            in_set[x] = True
            
    best_set = S.copy()
    best_weight = sum(weights[x] for x in S)
    
    def find_best_move(S, in_set, current_weight):
        best_gain = Fraction(0)
        best_move = None # (type, x, y)
        
        # Try additions
        for x in range(1, N + 1):
            if in_set[x]:
                continue
            if can_add(x, S, in_set):
                gain = weights[x]
                if gain > best_gain:
                    best_gain = gain
                    best_move = ('add', x, None)
        
        # Try swaps
        # For each x in S, remove it.
        # Then try adding each y not in S.
        for x in list(S):
            # Remove x temporarily
            S.remove(x)
            in_set[x] = False
            
            for y in range(1, N + 1):
                if y == x or in_set[y]:
                    continue
                if can_add(y, S, in_set):
                    gain = weights[y] - weights[x]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('swap', x, y)
            
            # Restore x
            S.add(x)
            in_set[x] = True
            
        return best_move, best_gain

    # Main Loop: Best-Improvement Local Search with Kicks
    stall_count = 0
    max_stall = 20
    
    while time.time() < deadline:
        move, gain = find_best_move(S, in_set, best_weight)
        
        if gain > 0:
            # Apply move
            if move[0] == 'add':
                x = move[1]
                S.add(x)
                in_set[x] = True
                best_weight += weights[x]
            elif move[0] == 'swap':
                x, y = move[1], move[2]
                S.remove(x)
                in_set[x] = False
                S.add(y)
                in_set[y] = True
                best_weight += weights[y] - weights[x]
            
            best_set = S.copy()
            stall_count = 0
        else:
            stall_count += 1
            if stall_count > max_stall:
                # Kick: Randomly remove 2 elements to break local optimum
                if len(S) > 2:
                    items = list(S)
                    random.shuffle(items)
                    x1, x2 = items[0], items[1]
                    S.remove(x1)
                    in_set[x1] = False
                    S.remove(x2)
                    in_set[x2] = False
                    best_weight -= weights[x1] + weights[x2]
                    stall_count = 0
                else:
                    break

    return list(best_set)
