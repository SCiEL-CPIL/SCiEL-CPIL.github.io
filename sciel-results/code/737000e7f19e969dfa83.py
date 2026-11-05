"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    deadline = time.time() + 85.0
    N = 256
    
    # Precompute conflict graph: two numbers conflict if they can form a 3-AP with a third number in [1,N]
    # Actually, for 3-AP-free sets, we need: no three distinct a,b,c with a+c=2b
    # Two elements x,y conflict if there exists z in [1,N] such that {x,y,z} forms a 3-AP
    # This means: z = 2x - y, or z = 2y - x, or z = (x+y)/2 (if x+y even)
    
    # Better approach: define adjacency where x~y if adding both to any set would create a 3-AP
    # Actually, the constraint is global: S is 3-AP-free iff no three elements form an AP.
    # For branch and bound on independent set: vertices are 1..N, edge between i,j if 
    # there exists k such that {i,j,k} is a 3-AP. But that's not quite right either.
    
    # The correct formulation: A set S is 3-AP-free iff for all distinct a,b,c in S, a+c != 2b.
    # This is NOT a pairwise constraint in the usual sense - it's a 3-wise constraint.
    # However, we can model it as: when building S incrementally, adding x to S is valid iff
    # for all a in S: (a+x)/2 not in S and a+x <= N (for x as largest), and x-a >= 0 (for x as smallest).
    
    # Let's use a different approach: greedy + local search with randomization, optimized with bitsets.
    
    # Precompute for each x, the set of y values that would form a 3-AP with some existing element
    # Actually, let's precompute: for each pair (a, x), what is the "forbidden" middle point
    # If we have a and x in S, then (a+x)/2 must NOT be in S (if integer and distinct from both)
    
    # For efficiency, let's use a bitset representation
    # N = 256, so we can use Python integers as bitsets (256 bits)
    
    # Precompute: for each x, list of pairs (a, b) such that a, b, x or a, x, b or x, a, b form 3-AP
    # i.e., x conflicts with having both a and b in S where {a,b,x} is a 3-AP
    
    # Actually, the cleanest approach: 
    # can_add(S, x) checks if adding x creates any 3-AP.
    # A 3-AP containing x: (x-d, x, x+d) or (x, x+d, x+2d) or (x-2d, x-d, x)
    # So for each d>=1: check if (x-d, x+d) both in S, or (x+d, x+2d) both in S, or (x-2d, x-d) both in S
    
    def can_add_bitset(bitset, x):
        """Check if adding x (1-indexed) to set represented by bitset creates a 3-AP.
        Bit i (0-indexed) represents element i+1."""
        # For each d >= 1:
        # Case 1: (x-d, x, x+d) - need both x-d and x+d in S
        # Case 2: (x, x+d, x+2d) - need both x+d and x+2d in S  
        # Case 3: (x-2d, x-d, x) - need both x-2d and x-d in S
        for d in range(1, N + 1):
            # Case 1: x-d and x+d
            a = x - d
            b = x + d
            if 1 <= a <= N and 1 <= b <= N:
                if (bitset >> (a - 1)) & 1 and (bitset >> (b - 1)) & 1:
                    return False
            # Case 2: x+d and x+2d
            a = x + d
            b = x + 2 * d
            if 1 <= a <= N and 1 <= b <= N:
                if (bitset >> (a - 1)) & 1 and (bitset >> (b - 1)) & 1:
                    return False
            # Case 3: x-2d and x-d
            a = x - 2 * d
            b = x - d
            if 1 <= a <= N and 1 <= b <= N:
                if (bitset >> (a - 1)) & 1 and (bitset >> (b - 1)) & 1:
                    return False
        return True
    
    def bitset_to_set(bitset):
        return [i + 1 for i in range(N) if (bitset >> i) & 1]
    
    def set_to_bitset(S):
        bits = 0
        for x in S:
            bits |= (1 << (x - 1))
        return bits
    
    def recip_sum_from_bitset(bitset):
        total = Fraction(0, 1)
        for i in range(N):
            if (bitset >> i) & 1:
                total += Fraction(1, i + 1)
        return total
    
    # Greedy initialization: add elements in order of 1/x (smallest first)
    def greedy_init(order=None):
        if order is None:
            order = list(range(1, N + 1))
        bitset = 0
        for x in order:
            if can_add_bitset(bitset, x):
                bitset |= (1 << (x - 1))
        return bitset
    
    # Local search: try swaps (remove one, add another)
    def local_search(bitset, deadline):
        best_bitset = bitset
        best_score = recip_sum_from_bitset(best_bitset)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = bitset_to_set(best_bitset)
            random.shuffle(S_list)
            for a in S_list:
                if time.time() > deadline - 1:
                    break
                trial = best_bitset & ~(1 << (a - 1))
                # Try adding each x not in trial
                for x in range(1, N + 1):
                    if (trial >> (x - 1)) & 1:
                        continue
                    if can_add_bitset(trial, x):
                        new_bitset = trial | (1 << (x - 1))
                        new_score = recip_sum_from_bitset(new_bitset)
                        if new_score > best_score:
                            best_bitset = new_bitset
                            best_score = new_score
                            improved = True
                            break
            # Try pure additions
            for x in range(1, N + 1):
                if time.time() > deadline - 1:
                    break
                if not ((best_bitset >> (x - 1)) & 1):
                    if can_add_bitset(best_bitset, x):
                        best_bitset |= (1 << (x - 1))
                        best_score = recip_sum_from_bitset(best_bitset)
                        improved = True
        return best_bitset
    
    # Branch and bound with bitset acceleration
    def branch_and_bound(bitset, remaining_bits, current_score, deadline):
        """Recursive B&B. remaining_bits is bitset of candidates not yet decided.
        current_score is Fraction sum of reciprocals of elements in bitset."""
        if time.time() > deadline:
            return bitset, current_score
        
        if remaining_bits == 0:
            return bitset, current_score
        
        # Upper bound: current_score + sum of 1/x for all remaining candidates
        # (this is loose but fast)
        upper_bound = current_score
        temp = remaining_bits
        while temp:
            i = (temp & -temp).bit_length() - 1
            upper_bound += Fraction(1, i + 1)
            temp &= temp - 1
        
        if upper_bound <= current_score:
            return bitset, current_score
        
        # Pick a candidate to branch on - choose the one with highest 1/x (smallest number)
        # among remaining
        temp = remaining_bits
        best_x = -1
        while temp:
            i = (temp & -temp).bit_length() - 1
            if best_x == -1 or (i + 1) < best_x:
                best_x = i + 1
            temp &= temp - 1
        
        if best_x == -1:
            return bitset, current_score
        
        x = best_x
        x_bit = 1 << (x - 1)
        new_remaining = remaining_bits & ~x_bit
        
        # Branch 1: include x
        if can_add_bitset(bitset, x):
            new_bitset = bitset | x_bit
            new_score = current_score + Fraction(1, x)
            result1, score1 = branch_and_bound(new_bitset, new_remaining, new_score, deadline)
        else:
            result1, score1 = bitset, current_score
        
        # Branch 2: exclude x
        result2, score2 = branch_and_bound(bitset, new_remaining, current_score, deadline)
        
        if score1 >= score2:
            return result1, score1
        else:
            return result2, score2
    
    # Run multiple randomized restarts
    best_bitset = None
    best_score = Fraction(0, 1)
    
    # First: deterministic greedy
    init_bits = greedy_init()
    init_bits = local_search(init_bits, deadline - 5)
    best_bitset = init_bits
    best_score = recip_sum_from_bitset(best_bitset)
    
    # Randomized restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 3:
            break
        
        # Random perturbation of the greedy order
        order = list(range(1, N + 1))
        random.shuffle(order)
        # Bias towards small numbers: sort by random key weighted towards small
        order.sort(key=lambda x: random.random() ** 2 * x)
        
        init_bits = greedy_init(order)
        init_bits = local_search(init_bits, deadline - 3)
        cur_score = recip_sum_from_bitset(init_bits)
        
        if cur_score > best_score:
            best_score = cur_score
            best_bitset = init_bits
    
    # Final polish on the best solution
    best_bitset = local_search(best_bitset, deadline - 1)
    best_score = recip_sum_from_bitset(best_bitset)
    
    return bitset_to_set(best_bitset)
# EVOLVE-BLOCK-END
