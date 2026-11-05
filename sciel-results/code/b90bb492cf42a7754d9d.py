import time
import random

def construct():
    deadline = time.time() + 85
    N = 180
    
    # Precompute forbidden triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c  <=> (b-a)(c-a) = a^2
    # For each a, find divisors d of a^2 such that d < a.
    # b = a + d, c = a + a^2/d
    # Check b < c and c <= N
    
    # conflict_pairs[x] will store a list of tuples (y, z)
    # such that {x, y, z} is a forbidden triple.
    # Note: The triple is undirected in the set context.
    # So if (a,b,c) is a triple, then:
    # conflict_pairs[a] contains (b,c)
    # conflict_pairs[b] contains (a,c)
    # conflict_pairs[c] contains (a,b)
    
    conflict_pairs = [list() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and c <= N:
                    # Triple (a, b, c)
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
    
    # Bitset representation for fast membership tests
    # bitset[x] is a bitmask where bit i is 1 if i is in the set
    # We use Python's arbitrary-precision integers as bitsets
    
    def make_bitset(S):
        bits = 0
        for x in S:
            bits |= (1 << x)
        return bits
    
    def bitset_has(bits, x):
        return (bits >> x) & 1
    
    def get_conflicts_bitset(x, bits):
        """
        Returns a list of pairs (y, z) from conflict_pairs[x]
        such that both y and z are in the bitset.
        """
        conflicts = []
        for y, z in conflict_pairs[x]:
            if bitset_has(bits, y) and bitset_has(bits, z):
                conflicts.append((y, z))
        return conflicts

    def can_add_bitset(x, bits):
        """Check if x can be added to the set represented by bits."""
        for y, z in conflict_pairs[x]:
            if bitset_has(bits, y) and bitset_has(bits, z):
                return False
        return True

    def init_solution(seed):
        """Create an initial solution using a given seed."""
        random.seed(seed)
        # Start with upper half, then greedily add
        S = set(range(91, N + 1))
        for x in range(1, 91):
            if can_add_bitset(x, make_bitset(S)):
                S.add(x)
        return S

    best_S = None
    best_size = 0
    
    # Try multiple initial seeds to find a good starting point
    for seed in [42, 123, 456, 789, 321]:
        S = init_solution(seed)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    all_nums = set(range(1, N + 1))
    
    # Local Search with 1-swap and 2-swap moves
    random.seed(42)
    
    # Use the best initial solution
    S = set(best_S)
    bits = make_bitset(S)
    
    for iteration in range(5000000):
        if time.time() > deadline:
            break
        
        excluded = all_nums - S
        if not excluded:
            break
            
        x = random.choice(tuple(excluded))
        conflicts = get_conflicts_bitset(x, bits)
        
        if not conflicts:
            S.add(x)
            bits |= (1 << x)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
            continue
        
        # Try to resolve conflicts by removing elements
        # Conflicts are pairs (y, z). We need to remove at least one from each pair.
        
        # Try removing 1 element
        candidates = set()
        for y, z in conflicts:
            candidates.add(y)
            candidates.add(z)
            
        found_swap = False
        cand_list = list(candidates)
        random.shuffle(cand_list)
        
        for u in cand_list:
            # Remove u, check if x becomes addable
            S.remove(u)
            bits ^= (1 << u)
            if can_add_bitset(x, bits):
                S.add(x)
                bits |= (1 << x)
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                found_swap = True
                break
            else:
                S.add(u)
                bits |= (1 << u)
        
        if found_swap:
            continue
        
        # Try 2-swap: remove u, v and add x, y
        # This maintains set size while exploring a larger neighborhood
        if random.random() < 0.15:
            # Find all elements that could be removed to help add x
            # Then find an element y that could be added after removing u,v
            if len(cand_list) >= 2:
                # Try a few random pairs (u, v) from candidates
                for _ in range(5):
                    u, v = random.sample(cand_list, 2)
                    # Remove u and v
                    S.remove(u)
                    S.remove(v)
                    bits ^= (1 << u)
                    bits ^= (1 << v)
                    
                    # Check if x can now be added
                    if can_add_bitset(x, bits):
                        S.add(x)
                        bits |= (1 << x)
                        
                        # Now try to add some element y from excluded to maintain size
                        # Look for an element y that was previously blocked
                        # but might now be addable
                        new_excluded = all_nums - S
                        if new_excluded:
                            # Try a few random elements
                            for y in random.sample(list(new_excluded), min(5, len(new_excluded))):
                                if can_add_bitset(y, bits):
                                    S.add(y)
                                    bits |= (1 << y)
                                    break
                        
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
                        found_swap = True
                        break
                    else:
                        S.add(u)
                        S.add(v)
                        bits |= (1 << u)
                        bits |= (1 << v)
                    
                    if found_swap:
                        break
        
        # Periodic restart from a different seed to escape deep local minima
        if iteration % 100000 == 0 and iteration > 0:
            new_seed = random.randint(1, 1000000)
            new_S = init_solution(new_seed)
            if len(new_S) > best_size:
                best_S = set(new_S)
                best_size = len(new_S)
                S = set(new_S)
                bits = make_bitset(S)

    return sorted(best_S)
