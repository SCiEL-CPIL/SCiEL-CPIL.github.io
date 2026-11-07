"""Combined program: greedy construction with targeted local search and polish."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def _can_add(S_list, x):
    """
    Check if x can be added to S (given as S_list).
    For every three distinct members a, b, c in S ∪ {x}, a does not divide b*c.
    """
    n = len(S_list)
    
    # 1. x divides a*b for distinct a,b in S
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if (a * b) % x == 0:
                return False
    
    # 2. a divides x*b for distinct a,b in S
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            if (x * b) % a == 0:
                return False
    
    return True

def _can_add_fast(S_list, x, pair_products, pair_indices):
    """
    Fast can_add using precomputed pair products.
    pair_products[i] = S_list[i] * S_list[j] for all i<j
    pair_indices[i] = (i, j)
    """
    n = len(S_list)
    
    # 1. x divides a*b for distinct a,b in S
    for prod, (i, j) in zip(pair_products, pair_indices):
        if prod % x == 0:
            return False
    
    # 2. a divides x*b for distinct a,b in S
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            if (x * b) % a == 0:
                return False
    
    return True

def _is_valid(S_list):
    """
    Check if S is a valid set.
    For every three distinct members a, b, c in S, a does not divide b*c.
    """
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            for k in range(n):
                if i == k or j == k:
                    continue
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _greedy_construct(order, deadline):
    """
    Greedily construct a set by iterating through 'order' and adding
    if the resulting set remains valid.
    """
    S = set()
    S_list = []
    
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(S_list, x):
            S.add(x)
            S_list.append(x)
            
    return S, S_list

def _compute_pair_data(S_list):
    """Compute pair products and indices for fast can_add."""
    n = len(S_list)
    pair_products = []
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_products.append(S_list[i] * S_list[j])
            pair_indices.append((i, j))
    return pair_products, pair_indices

def _find_blocking_pairs(S_list, x):
    """
    Find which pairs in S block x from being added.
    Returns list of (a, b) pairs that conflict with x.
    """
    blockers = []
    n = len(S_list)
    
    # 1. x divides a*b
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if (a * b) % x == 0:
                blockers.append((a, b))
    
    # 2. a divides x*b
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            if (x * b) % a == 0:
                blockers.append((a, b))
    
    return blockers

def _polish_pass(current_set, current_list, non_members, deadline):
    """
    Final high-precision polish: systematically try all adds, then all 1-for-1 swaps,
    then all 2-for-2 swaps until no improvement.
    """
    improved = True
    max_rounds = 50
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        if not improved:
            break
        improved = False
        
        # Try all adds
        for idx in range(len(non_members)):
            if time.time() > deadline:
                break
            x = non_members[idx]
            if _can_add(current_list, x):
                current_set.add(x)
                current_list.append(x)
                non_members.pop(idx)
                improved = True
                break
        
        if improved:
            continue
        
        # Try all 1-for-1 swaps
        if len(current_set) > 0:
            for i in range(len(current_list)):
                if time.time() > deadline:
                    break
                y = current_list[i]
                temp_list = [v for v in current_list if v != y]
                for j in range(len(non_members)):
                    if time.time() > deadline:
                        break
                    x = non_members[j]
                    if _can_add(temp_list, x):
                        current_set.remove(y)
                        current_set.add(x)
                        current_list = temp_list + [x]
                        non_members[j] = y
                        improved = True
                        break
                if improved:
                    break
        
        if improved:
            continue
        
        # Try all 2-for-2 swaps
        if len(current_set) >= 2 and len(non_members) >= 2:
            found = False
            for i in range(len(current_list)):
                if time.time() > deadline:
                    break
                y1 = current_list[i]
                for j in range(i + 1, len(current_list)):
                    if time.time() > deadline:
                        break
                    y2 = current_list[j]
                    temp_list = [v for v in current_list if v != y1 and v != y2]
                    for k in range(len(non_members)):
                        if time.time() > deadline:
                            break
                        x1 = non_members[k]
                        if _can_add(temp_list, x1):
                            temp_list2 = temp_list + [x1]
                            for m in range(k + 1, len(non_members)):
                                if time.time() > deadline:
                                    break
                                x2 = non_members[m]
                                if _can_add(temp_list2, x2):
                                    current_set.discard(y1)
                                    current_set.discard(y2)
                                    current_set.add(x1)
                                    current_set.add(x2)
                                    current_list = temp_list + [x1, x2]
                                    non_members[k] = y1
                                    non_members[m] = y2
                                    found = True
                                    break
                    if found:
                        break
                if found:
                    break
            if found:
                improved = True
    
    return current_set, current_list, non_members

def construct():
    deadline = time.time() + 55.0
    start_time = time.time()
    
    all_nums = list(range(2, 257))
    
    best_set = set()
    best_size = 0
    
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0: return False
        return True
            
    primes = [x for x in all_nums if is_prime(x)]
    composites = [x for x in all_nums if not is_prime(x)]
    
    orders = [
        list(range(2, 257)),
        list(range(256, 1, -1)),
        primes + composites,
        composites + primes,
    ]
    
    rng = random.Random(42)
    for _ in range(30):
        shuffled = all_nums.copy()
        rng.shuffle(shuffled)
        orders.append(shuffled)
        
    for order in orders:
        if time.time() > deadline:
            break
        S, S_list = _greedy_construct(order, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    current_set = best_set.copy()
    current_list = list(current_set)
    current_size = best_size
    
    non_members = [x for x in all_nums if x not in current_set]
    
    no_improve_count = 0
    max_no_improve = 300
    
    while time.time() < deadline - 5.0:
        if not non_members:
            break
        improved = False
        
        r = random.random()
        if r < 0.5:
            # Try targeted Add: pick a non-member that resolves most blockers
            # First, try a random add
            idx = random.randint(0, len(non_members) - 1)
            x = non_members[idx]
            if _can_add(current_list, x):
                current_set.add(x)
                current_list.append(x)
                current_size += 1
                non_members.pop(idx)
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set.copy()
                improved = True
                no_improve_count = 0
            else:
                # Targeted: find a non-member that resolves blockers
                # Sample a few candidates and pick the one with fewest conflicts
                candidates = random.sample(non_members, min(10, len(non_members)))
                best_candidate = None
                min_conflicts = float('inf')
                for cand in candidates:
                    if _can_add(current_list, cand):
                        best_candidate = cand
                        min_conflicts = 0
                        break
                    else:
                        # Count conflicts (approximate)
                        n_conf = 0
                        for a in current_list:
                            for b in current_list:
                                if a == b:
                                    continue
                                if (cand * b) % a == 0:
                                    n_conf += 1
                                    break
                        if n_conf < min_conflicts:
                            min_conflicts = n_conf
                            best_candidate = cand
                
                if best_candidate is not None:
                    # Try the best candidate
                    if _can_add(current_list, best_candidate):
                        idx2 = non_members.index(best_candidate)
                        current_set.add(best_candidate)
                        current_list.append(best_candidate)
                        current_size += 1
                        non_members.pop(idx2)
                        if current_size > best_size:
                            best_size = current_size
                            best_set = current_set.copy()
                        improved = True
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
        elif r < 0.8:
            # 1-for-1 Swap
            if len(current_set) > 0:
                y = random.choice(current_list)
                idx = random.randint(0, len(non_members) - 1)
                x = non_members[idx]
                temp_list = [v for v in current_list if v != y]
                if _can_add(temp_list, x):
                    current_set.remove(y)
                    current_set.add(x)
                    current_list = temp_list + [x]
                    non_members[idx] = y
                    no_improve_count = 0
                    improved = True
                else:
                    no_improve_count += 1
        else:
            # 2-for-2 Swap: remove two, add two
            if len(current_set) >= 2 and len(non_members) >= 2:
                y1, y2 = random.sample(current_list, 2)
                idx1, idx2 = random.sample(range(len(non_members)), 2)
                x1, x2 = non_members[idx1], non_members[idx2]
                temp_list = [v for v in current_list if v != y1 and v != y2]
                if _can_add(temp_list, x1) and _can_add(temp_list + [x1], x2):
                    current_set.discard(y1)
                    current_set.discard(y2)
                    current_set.add(x1)
                    current_set.add(x2)
                    current_list = temp_list + [x1, x2]
                    non_members[idx1] = y1
                    non_members[idx2] = y2
                    no_improve_count = 0
                    improved = True
                else:
                    no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            if random.random() < 0.5:
                current_set = best_set.copy()
                current_list = list(current_set)
                current_size = best_size
                non_members = [x for x in all_nums if x not in current_set]
    
    # Final polish pass
    if time.time() < deadline - 2.0:
        current_set, current_list, non_members = _polish_pass(
            current_set, current_list, non_members, deadline - 2.0
        )
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
    
    result = list(best_set)
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
