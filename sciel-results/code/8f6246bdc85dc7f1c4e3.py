"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55  # safety margin before 90s limit
    TARGET = 200
    ALL = list(range(1, 101))
    ALL_SET = set(ALL)
    
    def _build_reachability(S):
        """Build reachable-sums bitset for set S using numpy."""
        import numpy as np
        reach = np.zeros(TARGET + 1, dtype=bool)
        reach[0] = True
        for x in S:
            if x >= TARGET:
                continue
            # shift reach right by x, mask to valid range
            shifted = reach[:TARGET - x + 1].copy()
            reach[x:] |= shifted
            if reach[TARGET]:
                return reach  # already hit target
        return reach
    
    def _can_add(S, x):
        """Check if adding x to S keeps it valid (no subset sums to TARGET)."""
        import numpy as np
        reach = _build_reachability(S)
        if reach[TARGET - x]:
            return False
        return True
    
    def greedy(order):
        """Greedy construction based on a given order of elements."""
        S = []
        S_set = set()
        import numpy as np
        reach = np.zeros(TARGET + 1, dtype=bool)
        reach[0] = True
        
        for x in order:
            if x >= TARGET:
                S.append(x)
                S_set.add(x)
                continue
            if reach[TARGET - x]:
                continue
            # Update reachability
            shifted = reach[:TARGET - x + 1].copy()
            reach[x:] |= shifted
            S.append(x)
            S_set.add(x)
            
        return S_set

    def local_improve(S_set):
        """Try to add excluded elements to S_set if possible. Uses fast reachability."""
        import numpy as np
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            missing = [i for i in ALL if i not in S_set]
            for x in sorted(missing, reverse=True):
                if time.time() > deadline - 5:
                    break
                if _can_add(S_set, x):
                    S_set.add(x)
                    improved = True
                    break  # restart scan
        return S_set

    def local_improve_swaps(S_set):
        """Simulated-annealing-style: try swaps to escape local optima."""
        import numpy as np
        best_set = set(S_set)
        best_size = len(best_set)
        T = 1.0
        cooling = 0.97
        
        for _ in range(50):
            if time.time() > deadline - 8:
                break
            # Try to add a random missing element
            missing = [i for i in ALL if i not in best_set]
            if missing:
                x = random.choice(missing)
                if _can_add(best_set, x):
                    best_set.add(x)
                    best_size = len(best_set)
                    T = max(T * 0.9, 0.01)
                    continue
            
            # Try a swap: remove one element, add another
            if len(best_set) > 1:
                to_remove = random.choice(list(best_set))
                candidates = [i for i in ALL if i not in best_set and i != to_remove]
                if candidates:
                    to_add = random.choice(candidates)
                    trial = set(best_set)
                    trial.discard(to_remove)
                    trial.add(to_add)
                    # Check validity
                    if _can_add(trial - {to_add}, to_add):
                        # Valid swap
                        if len(trial) > best_size:
                            best_set = trial
                            best_size = len(trial)
                            T = max(T * 0.9, 0.01)
                        elif random.random() < T:
                            # Accept worsening move with probability T
                            best_set = trial
                    else:
                        # Try without the swap
                        pass
            
            T = max(T * cooling, 0.01)
        
        return best_set

    best_set = set()
    best_len = 0
    
    # 1. Static orders
    static_orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(1, 101, 2)),  # odd numbers
        list(range(2, 101, 2)),  # even numbers
    ]
    
    for o in static_orders:
        if time.time() > deadline - 15:
            break
        S = greedy(o)
        S = local_improve(S)
        if len(S) > best_len:
            best_len = len(S)
            best_set = S
        if best_len == 100:
            return list(best_set)

    # 2. Adaptive random search with priorities
    priorities = {x: 1.0 for x in ALL}
    
    # Run as many iterations as possible until deadline
    while time.time() < deadline - 10:
        if random.random() < 0.15:
            order = ALL[:]
            random.shuffle(order)
        else:
            order = sorted(ALL, key=lambda x: (priorities[x] + random.uniform(0, 0.1), x), reverse=True)
        
        S = greedy(order)
        S = local_improve(S)
        
        if len(S) > best_len:
            best_len = len(S)
            best_set = S
            if best_len == 100:
                return list(best_set)
        
        # Update priorities
        for x in ALL:
            if x in S:
                priorities[x] *= 0.95
            else:
                priorities[x] *= 1.05
                
    # Final local improvement + swap search on the best set found
    if best_set:
        best_set = local_improve(best_set)
        if time.time() < deadline - 3:
            best_set = local_improve_swaps(best_set)
            best_set = local_improve(best_set)

    if not best_set:
        return [100]
        
    return list(best_set)
# EVOLVE-BLOCK-END
