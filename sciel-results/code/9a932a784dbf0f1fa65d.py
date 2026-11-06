import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 88.0

    moduli = list(range(2, 129))

    # Precompute gcds
    gcd_val = [[0] * 129 for _ in range(129)]
    for i in range(2, 129):
        for j in range(i, 129):
            gcd_val[i][j] = math.gcd(i, j)
            gcd_val[j][i] = gcd_val[i][j]

    def try_add(m, current_set):
        """
        current_set: list of (r, m) tuples.
        Returns a valid residue r for m if possible, else None.
        Uses forbidden array for efficiency.
        """
        if not current_set:
            return 0

        # Check for coprime moduli: if any existing modulus is coprime to m,
        # the classes MUST intersect, so we cannot add m.
        for r_j, m_j in current_set:
            if gcd_val[m][m_j] == 1:
                return None

        forbidden = [False] * m
        
        for r_j, m_j in current_set:
            g = gcd_val[m][m_j]
            # g > 1 is guaranteed here
            target = r_j % g
            for r in range(target, m, g):
                forbidden[r] = True
        
        for r in range(m):
            if not forbidden[r]:
                return r
        return None

    def greedy(order):
        current_set = []
        for m in order:
            r = try_add(m, current_set)
            if r is not None:
                current_set.append((r, m))
        return current_set

    best_set = []
    best_size = 0

    def count_divisors(n):
        cnt = 0
        for i in range(2, n + 1):
            if n % i == 0:
                cnt += 1
        return cnt

    # Heuristic: Prioritize even numbers to ensure pairwise non-coprime property
    # and thus allow a large set.
    even_moduli = [m for m in moduli if m % 2 == 0]
    odd_moduli = [m for m in moduli if m % 2 != 0]
    
    heuristic_orders = [
        sorted(even_moduli, key=lambda x: -count_divisors(x)) + sorted(odd_moduli, key=lambda x: -count_divisors(x)),
        sorted(even_moduli, reverse=True) + sorted(odd_moduli, reverse=True),
        sorted(moduli, key=lambda x: -count_divisors(x)),
        sorted(moduli, reverse=True),
    ]

    for order in heuristic_orders:
        if time.time() > deadline:
            break
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Random restarts
    num_restarts = 20000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        random_order = moduli.copy()
        random.shuffle(random_order)
        s = greedy(random_order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Improved Local Search: Try to add missing moduli.
    # If direct addition fails, try to swap residues of existing moduli to make room.
    def local_search(current_set, time_limit):
        current_set = list(current_set)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            current_moduli = {m for r, m in current_set}
            
            # Find a missing modulus that is non-coprime with all current moduli
            candidates = [m for m in moduli if m not in current_moduli and all(gcd_val[m][m_j] > 1 for r_j, m_j in current_set)]
            
            for m_new in candidates:
                r_new = try_add(m_new, current_set)
                if r_new is not None:
                    current_set.append((r_new, m_new))
                    improved = True
                    break
                
                # If direct add failed, try to change one existing residue to free up space
                # Only try this if the set is small enough to be fast
                if len(current_set) < 40:
                    for idx, (r_old, m_old) in enumerate(current_set):
                        # Try a few random residues for m_old
                        for _ in range(5):
                            r_try = random.randint(0, m_old - 1)
                            if r_try == r_old:
                                continue
                            # Check if r_try is valid for m_old against others
                            valid = True
                            for r_j, m_j in current_set:
                                if m_j == m_old:
                                    continue
                                g = gcd_val[m_old][m_j]
                                if (r_try - r_j) % g == 0:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            old_pair = current_set[idx]
                            current_set[idx] = (r_try, m_old)
                            r_new2 = try_add(m_new, current_set)
                            if r_new2 is not None:
                                current_set.append((r_new2, m_new))
                                improved = True
                                break
                            current_set[idx] = old_pair
                        if improved:
                            break
            if not improved:
                break
        return current_set

    if best_set and time.time() < deadline - 5:
        improved_set = local_search(best_set, deadline - 2)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)

    result = [[r, m] for r, m in best_set]

    if not result:
        result = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]

    return result
