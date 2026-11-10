import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 512
    nodes = list(range(1, n + 1))
    
    # Precompute conflict matrix
    # conflict[i] is a set of j such that lcm(i, j) > 512
    # Using a list of lists of booleans or just checking on fly?
    # 512*512 is small. Let's use a 2D array or list of sets.
    # For fast "is x in conflicts of c?", we can use a bitset (int) or a boolean array.
    # Python ints can be large, so we can use an integer as a bitset for each i.
    # Bit j is 1 if j conflicts with i.
    
    conflict_bits = [0] * (n + 1)
    for i in range(1, n + 1):
        bits = 0
        for j in range(1, n + 1):
            if i == j:
                continue
            g = gcd(i, j)
            l = (i // g) * j
            if l > n:
                bits |= (1 << (j - 1))
        conflict_bits[i] = bits
    
    def get_conflict_count(x, S):
        # Count how many elements in S conflict with x
        # S is a set of integers
        count = 0
        bits_x = conflict_bits[x]
        for c in S:
            if bits_x & (1 << (c - 1)):
                count += 1
        return count

    def is_compatible(x, S):
        bits_x = conflict_bits[x]
        for c in S:
            if bits_x & (1 << (c - 1)):
                return False
        return True

    def generate_seed(seed_type):
        if seed_type == 0:
            # Divisors of 504
            return [k for k in range(1, n + 1) if 504 % k == 0]
        elif seed_type == 1:
            # Divisors of 480
            return [k for k in range(1, n + 1) if 480 % k == 0]
        elif seed_type == 2:
            # Divisors of 420
            return [k for k in range(1, n + 1) if 420 % k == 0]
        elif seed_type == 3:
            # Divisors of 360
            return [k for k in range(1, n + 1) if 360 % k == 0]
        else:
            # Greedy random
            S = set()
            remaining = list(nodes)
            random.shuffle(remaining)
            for x in remaining:
                if is_compatible(x, S):
                    S.add(x)
            return list(S)

    def bron_kerbosch(R, P, X):
        """
        Bron-Kerbosch with pivot for maximum clique.
        R: current clique (list)
        P: candidates that can be added (list)
        X: excluded candidates (list)
        Returns best clique found.
        """
        if not P and not X:
            return list(R)
        
        # Choose pivot u from P ∪ X to minimize branching
        # u should maximize |P ∩ N(u)|
        best_u = None
        best_count = -1
        for u in P + X:
            count = 0
            bits_u = conflict_bits[u]
            for v in P:
                if not (bits_u & (1 << (v - 1))):
                    count += 1
            if count > best_count:
                best_count = count
                best_u = u
        
        if best_u is None:
            return list(R)
        
        # Branch on P \ N(best_u)
        bits_best_u = conflict_bits[best_u]
        for v in list(P):
            if bits_best_u & (1 << (v - 1)):
                continue  # v is neighbor of best_u, skip
            
            # Add v to R, recurse
            R.append(v)
            # New P = P ∩ N(v), New X = X ∩ N(v)
            bits_v = conflict_bits[v]
            new_P = [w for w in P if w != v and not (bits_v & (1 << (w - 1)))]
            new_X = [w for w in X if not (bits_v & (1 << (w - 1)))]
            
            result = bron_kerbosch(R, new_P, new_X)
            if len(result) > best_size[0]:
                best_size[0] = len(result)
                best_clique[0] = list(result)
            
            R.pop()
            P.remove(v)
            X.append(v)
        
        return best_clique[0]

    # First, try Bron-Kerbosch for maximum clique
    # Start with all nodes as candidates
    best_size = [0]
    best_clique = [[]]
    
    # Use a simple heuristic: start with a good seed
    # The divisors of 504 give a large compatible set
    seed0 = [k for k in range(1, n + 1) if 504 % k == 0]
    if seed0:
        best_size[0] = len(seed0)
        best_clique[0] = seed0
    
    # Now run Bron-Kerbosch from scratch
    # This might be slow, so we limit it
    bk_start = time.time()
    bk_deadline = bk_start + 20.0  # Give 20 seconds to Bron-Kerbosch
    
    # We need to modify bron_kerbosch to check time
    def bron_kerbosch_timed(R, P, X):
        """
        Bron-Kerbosch with pivot for maximum clique, with time limit.
        """
        if time.time() > bk_deadline:
            return
        
        if not P and not X:
            if len(R) > best_size[0]:
                best_size[0] = len(R)
                best_clique[0] = list(R)
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        best_u = None
        best_count = -1
        for u in P + X:
            count = 0
            bits_u = conflict_bits[u]
            for v in P:
                if not (bits_u & (1 << (v - 1))):
                    count += 1
            if count > best_count:
                best_count = count
                best_u = u
        
        if best_u is None:
            return
        
        # Branch on P \ N(best_u)
        bits_best_u = conflict_bits[best_u]
        for v in list(P):
            if time.time() > bk_deadline:
                return
            if bits_best_u & (1 << (v - 1)):
                continue  # v is neighbor of best_u, skip
            
            # Add v to R, recurse
            R.append(v)
            # New P = P ∩ N(v), New X = X ∩ N(v)
            bits_v = conflict_bits[v]
            new_P = [w for w in P if w != v and not (bits_v & (1 << (w - 1)))]
            new_X = [w for w in X if not (bits_v & (1 << (w - 1)))]
            
            bron_kerbosch_timed(R, new_P, new_X)
            
            R.pop()
            P.remove(v)
            X.append(v)
    
    # Run Bron-Kerbosch
    initial_P = list(range(1, n + 1))
    initial_X = []
    initial_R = []
    bron_kerbosch_timed(initial_R, initial_P, initial_X)
    
    # After Bron-Kerbosch, do a final greedy polish
    if best_clique[0]:
        S = set(best_clique[0])
        # Try to add any remaining compatible elements
        for x in range(1, n + 1):
            if x not in S and is_compatible(x, S):
                S.add(x)
        best_clique[0] = list(S)
        best_size[0] = len(S)
    
    # If Bron-Kerbosch didn't find anything good, fall back to simulated annealing
    if best_size[0] < 10:
        # Simulated annealing fallback
        seeds = [generate_seed(0), generate_seed(1), generate_seed(2), generate_seed(3), generate_seed(4)]
        
        for seed in seeds:
            if time.time() > deadline:
                break
                
            S = set(seed)
            current_size = len(S)
            
            # Maintain conflict counts for all x not in S
            conf_count = [0] * (n + 1)
            for x in range(1, n + 1):
                if x not in S:
                    conf_count[x] = get_conflict_count(x, S)
            
            temp = 100.0
            min_temp = 0.001
            cooling = 0.999
            
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_clique[0] = list(S)
            
            steps = 0
            while time.time() < deadline and temp > min_temp:
                added = False
                for _ in range(10):
                    x = random.randint(1, n)
                    if x not in S and conf_count[x] == 0:
                        S.add(x)
                        current_size += 1
                        bits_x = conflict_bits[x]
                        for y in range(1, n + 1):
                            if y not in S and bits_x & (1 << (y - 1)):
                                conf_count[y] += 1
                        if current_size > best_size[0]:
                            best_size[0] = current_size
                            best_clique[0] = list(S)
                        added = True
                        break
                
                if added:
                    continue
                
                swapped = False
                c = random.choice(list(S))
                for _ in range(10):
                    x = random.randint(1, n)
                    if x in S:
                        continue
                    
                    conflicts_c = (conflict_bits[x] & (1 << (c - 1))) != 0
                    if conflicts_c:
                        remaining_conflicts = conf_count[x] - 1
                    else:
                        remaining_conflicts = conf_count[x]
                    
                    if remaining_conflicts == 0:
                        if random.random() < 0.5:
                            S.remove(c)
                            S.add(x)
                            
                            bits_c = conflict_bits[c]
                            bits_x = conflict_bits[x]
                            
                            for y in range(1, n + 1):
                                if y in S:
                                    continue
                                if bits_c & (1 << (y - 1)):
                                    conf_count[y] -= 1
                                if bits_x & (1 << (y - 1)):
                                    conf_count[y] += 1
                            swapped = True
                            break
                
                if swapped:
                    pass

                temp *= cooling
                steps += 1
                if steps % 1000 == 0 and time.time() > deadline - 2.0:
                    break
    
    # Final greedy polish on the best result
    if best_clique[0]:
        S = set(best_clique[0])
        for x in range(1, n + 1):
            if x not in S and is_compatible(x, S):
                S.add(x)
        best_clique[0] = list(S)
    
    return best_clique[0] if best_clique[0] else [1]
