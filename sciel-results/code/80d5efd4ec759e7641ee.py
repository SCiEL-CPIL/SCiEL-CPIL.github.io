import time
import random
from math import gcd
from collections import defaultdict

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s limit
    
    def is_bad_triple(a, b, c):
        """Check if three elements form a bad triple (all pairwise gcds equal)."""
        return gcd(a, b) == gcd(a, c) == gcd(b, c)
    
    # Precompute gcd matrix for [1,128]
    gcd_mat = [[0]*129 for _ in range(129)]
    for i in range(1, 129):
        for j in range(i, 129):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    def count_conflicts(x, selected):
        """Count how many pairs (a,b) in selected form a bad triple with x."""
        count = 0
        for a in selected:
            for b in selected:
                if a >= b:
                    continue
                if gcd_mat[x][a] == gcd_mat[x][b] == gcd_mat[a][b]:
                    count += 1
        return count
    
    def can_add(x, selected):
        """Check if x can be added to selected without forming a bad triple."""
        for a in selected:
            for b in selected:
                if a >= b:
                    continue
                if gcd_mat[x][a] == gcd_mat[x][b] == gcd_mat[a][b]:
                    return False
        return True
    
    def greedy_build(order):
        """Build a valid set by adding elements in the given order."""
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected
    
    def local_search_add(selected, max_iters=200):
        """Try to add elements to the set."""
        selected = set(selected)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            excluded = [x for x in range(1, 129) if x not in selected]
            random.shuffle(excluded)
            for x in excluded:
                if can_add(x, selected):
                    selected.add(x)
                    improved = True
                    break
            if not improved:
                break
        return selected
    
    def simulated_annealing(initial_set, max_iters=5000):
        """Simulated annealing with swap moves."""
        current = set(initial_set)
        best = set(current)
        best_size = len(current)
        
        T = 2.0
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random element to remove
            if not current:
                break
            remove_x = random.choice(list(current))
            
            # Pick a random element to add
            excluded = [x for x in range(1, 129) if x not in current]
            if not excluded:
                break
            add_x = random.choice(excluded)
            
            if remove_x == add_x:
                continue
            
            # Try the swap
            new_set = (current - {remove_x}) | {add_x}
            
            # Check validity
            valid = True
            new_list = list(new_set)
            for i in range(len(new_list)):
                if not valid:
                    break
                for j in range(i+1, len(new_list)):
                    if not valid:
                        break
                    for k in range(j+1, len(new_list)):
                        if is_bad_triple(new_list[i], new_list[j], new_list[k]):
                            valid = False
                            break
            
            if not valid:
                continue
            
            # Size change: swap keeps same size, so only accept if we find a way to grow
            # Actually, let's try: remove one, then try to add up to 2
            temp = current - {remove_x}
            added = []
            for y in excluded:
                if len(added) >= 2:
                    break
                if can_add(y, temp | set(added)):
                    added.append(y)
            
            new_set = temp | set(added)
            delta = len(new_set) - len(current)
            
            if delta >= 0:
                current = new_set
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
            else:
                # Accept with probability
                if random.random() < (10 ** (delta / T)):
                    current = new_set
            
            T *= cooling
        
        return best
    
    # Strategy 1: Start with primes + powers of 2
    primes = [x for x in range(2, 129) if all(x % p != 0 for p in range(2, int(x**0.5)+1))]
    powers_of_2 = [2**i for i in range(7) if 2**i <= 128]  # 1,2,4,8,16,32,64,128
    
    initial = set(primes) | set(powers_of_2)
    # Remove 1 from powers_of_2 since 1 is in [1,128] and gcd(1,x)=1 for all x
    # Actually 1 is fine, let's keep it
    initial.add(1)
    
    best_set = local_search_add(initial, max_iters=500)
    best_size = len(best_set)
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Strategy 2: Simulated annealing from the best set
    sa_result = simulated_annealing(best_set, max_iters=3000)
    if len(sa_result) > best_size:
        best_size = len(sa_result)
        best_set = sa_result
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Strategy 3: Random restarts with different initial sets
    random.seed(42)
    for trial in range(30):
        if time.time() > deadline:
            break
        
        # Random initial set: primes + random evens + random odds
        init = set(primes)
        # Add some random numbers
        candidates = [x for x in range(1, 129) if x not in init]
        random.shuffle(candidates)
        for x in candidates[:20]:
            if can_add(x, init):
                init.add(x)
        
        s = local_search_add(init, max_iters=300)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
        
        # Try SA from this
        if time.time() < deadline - 10:
            sa_res = simulated_annealing(s, max_iters=1000)
            if len(sa_res) > best_size:
                best_size = len(sa_res)
                best_set = sa_res
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Strategy 4: More aggressive local search - try removing elements that block many additions
    current = set(best_set)
    for _ in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        # For each element in current, check how many excluded elements it blocks
        excluded = [x for x in range(1, 129) if x not in current]
        
        best_remove = None
        best_gain = 0
        
        for x in list(current):
            temp = current - {x}
            # Count how many excluded elements can now be added
            gain = 0
            for y in excluded:
                if can_add(y, temp):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_remove = x
        
        if best_remove is not None and best_gain > 0:
            current = current - {best_remove}
            # Try to add back as many as possible
            for y in excluded:
                if can_add(y, current):
                    current.add(y)
                    excluded.remove(y)
            if len(current) > best_size:
                best_size = len(current)
                best_set = current
                improved = True
        
        if not improved:
            break
    
    if not best_set:
        best_set = {1}
    
    return sorted(best_set)
