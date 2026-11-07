import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit

    # Precompute: for each pair (a,b) with a<b, the pair sums a+b, a+c, b+c
    # A hyperedge is (a,b,c) where a<b<c and a+b, a+c, b+c are all in S
    # We need to check: for distinct a,b,c in S, are a+b, a+c, b+c all in S?

    def count_violations(S_mask):
        """Count the number of forbidden triples (a,b,c) in S where all pair sums are also in S."""
        # Extract set elements
        S = []
        temp = S_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            S.append(idx)
            temp -= lsb
        
        S_set = set(S)
        violations = 0
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                b = S[j]
                for k in range(j + 1, n):
                    c = S[k]
                    if (a + b) in S_set and (a + c) in S_set and (b + c) in S_set:
                        violations += 1
        return violations

    def can_add(S_mask, x):
        """
        Check if adding x to S_mask results in a valid set.
        """
        # Case 1: x is a member of the triple (a=x)
        V_mask = S_mask & (S_mask >> x)
        
        if V_mask.bit_count() < 2:
            pass
        else:
            v_list = []
            temp = V_mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                v_list.append(idx)
                temp -= lsb
                
            n_v = len(v_list)
            if n_v >= 2:
                for i in range(n_v):
                    b = v_list[i]
                    for j in range(i + 1, n_v):
                        c = v_list[j]
                        s = b + c
                        if s == x or ((S_mask >> s) & 1):
                            return False
        
        # Case 2: x is a sum of two members (a+b=x)
        for a in range(1, x // 2 + 1):
            if ((S_mask >> a) & 1) == 0:
                continue
            b = x - a
            if b == a:
                continue
            if ((S_mask >> b) & 1) == 0:
                continue
            
            U_mask = (S_mask & (S_mask >> a)) & (S_mask & (S_mask >> b))
            U_mask &= ~(1 << a)
            U_mask &= ~(1 << b)
            
            if U_mask != 0:
                return False
        
        return True

    def greedy_construct(order, S_mask):
        """Greedy construction given a fixed order of candidates."""
        for x in order:
            if time.time() > deadline:
                break
            if can_add(S_mask, x):
                S_mask |= (1 << x)
        return S_mask

    def local_search(S_mask, max_passes=50):
        """Improve S via additions and systematic swaps."""
        S_set = set()
        temp = S_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            S_set.add(idx)
            temp -= lsb
            
        not_in_S = [x for x in range(1, 129) if x not in S_set]
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            
            improved = False
            random.shuffle(not_in_S)
            
            for x in not_in_S[:]:
                if time.time() > deadline:
                    break
                
                # Try to add x directly
                if can_add(S_mask, x):
                    S_mask |= (1 << x)
                    S_set.add(x)
                    not_in_S.remove(x)
                    improved = True
                    continue
                
                # Try swaps: remove y from S, add x
                S_list = list(S_set)
                random.shuffle(S_list)
                
                found_swap = False
                for y in S_list[:30]:
                    if time.time() > deadline:
                        break
                    S_temp_mask = S_mask ^ (1 << y)
                    if can_add(S_temp_mask, x):
                        S_mask = S_temp_mask | (1 << x)
                        S_set.remove(y)
                        S_set.add(x)
                        not_in_S.append(y)
                        if x in not_in_S:
                            not_in_S.remove(x)
                        found_swap = True
                        improved = True
                        break
                
                if found_swap:
                    break
            
            if not improved:
                break
        
        return S_mask

    def simulated_annealing(S_mask, max_iters=5000, T_start=10.0, T_end=0.01):
        """Simulated annealing to escape local optima using violation count as energy."""
        if time.time() > deadline:
            return S_mask
        
        # Build S set
        S_set = set()
        temp = S_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            S_set.add(idx)
            temp -= lsb
        
        not_in_S = [x for x in range(1, 129) if x not in S_set]
        
        # Initial energy
        current_energy = count_violations(S_mask)
        best_mask = S_mask
        best_energy = current_energy
        
        n_iters = max_iters
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / max(1, n_iters - 1))
        
        for it in range(n_iters):
            if time.time() > deadline:
                break
            
            T *= cooling
            
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3 and not_in_S:
                # Add a random element
                x = random.choice(not_in_S)
                new_mask = S_mask | (1 << x)
                new_energy = count_violations(new_mask)
                delta_e = new_energy - current_energy
                
                if delta_e <= 0 or random.random() < (2.718281828 ** (-delta_e / max(T, 1e-10))):
                    S_mask = new_mask
                    S_set.add(x)
                    not_in_S.remove(x)
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_mask = S_mask
                        
            elif move_type < 0.6 and S_set:
                # Remove a random element
                y = random.choice(list(S_set))
                new_mask = S_mask ^ (1 << y)
                new_energy = count_violations(new_mask)
                delta_e = new_energy - current_energy
                
                if delta_e <= 0 or random.random() < (2.718281828 ** (-delta_e / max(T, 1e-10))):
                    S_mask = new_mask
                    S_set.remove(y)
                    not_in_S.append(y)
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_mask = S_mask
                        
            else:
                # Swap: remove y from S, add x
                if S_set and not_in_S:
                    x = random.choice(not_in_S)
                    y = random.choice(list(S_set))
                    new_mask = (S_mask ^ (1 << y)) | (1 << x)
                    new_energy = count_violations(new_mask)
                    delta_e = new_energy - current_energy
                    
                    if delta_e <= 0 or random.random() < (2.718281828 ** (-delta_e / max(T, 1e-10))):
                        S_mask = new_mask
                        S_set.remove(y)
                        S_set.add(x)
                        not_in_S.remove(x)
                        not_in_S.append(y)
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_mask = S_mask
        
        # After SA, try to add any elements that can be added without violation
        for x in range(1, 129):
            if (best_mask >> x) & 1:
                continue
            if can_add(best_mask, x):
                best_mask |= (1 << x)
        
        return best_mask

    # Phase 1: Multiple greedy constructions with diverse orderings
    random.seed(42)
    
    best_mask = 0
    
    # Ascending
    S_mask = greedy_construct(list(range(1, 129)), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask
    
    # Descending
    S_mask = greedy_construct(list(range(128, 0, -1)), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask
    
    # Several random orders
    for i in range(50):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S_mask = greedy_construct(candidates, 0)
        if S_mask.bit_count() > best_mask.bit_count():
            best_mask = S_mask
    
    # Phase 2: Local search from best set
    if time.time() < deadline:
        best_mask = local_search(best_mask, max_passes=30)
    
    # Phase 3: Simulated annealing from best set
    if time.time() < deadline:
        sa_mask = simulated_annealing(best_mask, max_iters=3000, T_start=5.0, T_end=0.01)
        if sa_mask.bit_count() > best_mask.bit_count():
            best_mask = sa_mask
    
    # Phase 4: More random restarts with local search
    for i in range(50):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S_mask = greedy_construct(candidates, 0)
        S_mask = local_search(S_mask, max_passes=20)
        if S_mask.bit_count() > best_mask.bit_count():
            best_mask = S_mask
    
    # Phase 5: Final simulated annealing polish
    if time.time() < deadline and best_mask.bit_count() < 128:
        sa_mask = simulated_annealing(best_mask, max_iters=2000, T_start=3.0, T_end=0.01)
        if sa_mask.bit_count() > best_mask.bit_count():
            best_mask = sa_mask
    
    # Final: try to add any remaining elements
    if time.time() < deadline:
        for x in range(1, 129):
            if (best_mask >> x) & 1:
                continue
            if can_add(best_mask, x):
                best_mask |= (1 << x)
    
    # One more local search pass on the final best set
    if time.time() < deadline and best_mask.bit_count() < 128:
        best_mask = local_search(best_mask, max_passes=10)
    
    # Convert mask to list of ints
    result = []
    temp = best_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(int(idx))
        temp -= lsb
    
    return result
