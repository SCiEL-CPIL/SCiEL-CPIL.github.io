import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 192
    rng = random.Random(42)
    
    # Precompute conflict bitsets
    # conflict[x] = bitset of elements y such that x divides sum of any 3 distinct others including y
    # More precisely: for each candidate x, precompute which triples (a,b,c) from [1..n]\{x} have x | (a+b+c)
    # Then when checking if x can be added to set S, we need: no triple from S has x dividing its sum
    # And for each a in S, no pair (b,c) from S\{a} has a | (x+b+c)
    
    # Precompute for each x: the set of triples (a,b,c) with a<b<c, a,b,c != x, such that x | (a+b+c)
    # Represent as: for each x, a list of (a,b,c) triples
    # But this is too many. Instead, use a different approach.
    
    # Key insight: precompute for each pair (x, a) where x != a:
    #   For x to not divide any triple sum containing a: 
    #   Actually, let's think differently.
    
    # For a set S and candidate x:
    # Condition 1: For all triples (a,b,c) in C(S,3), x does NOT divide (a+b+c)
    # Condition 2: For all a in S, for all pairs (b,c) in C(S\{a},2), a does NOT divide (x+b+c)
    
    # Precompute:
    # triple_sum_mod[x][a][b] = (a+b) % x  -- too much memory
    # 
    # Alternative: For each x, precompute a bitset conflict_x where bit y is set if
    # there exist distinct a,b in [1..n]\{x,y} such that x | (y+a+b)
    # This tells us: if x is in the set, then y cannot be in the set (because a,b might also be)
    # But this is too strong - it's only a conflict if a AND b are both in the set.
    
    # Let's use a different representation:
    # For each x, precompute: pairs_conflict[x] = set of (a,b) pairs such that x | (a+b)
    # No wait, we need triples.
    
    # Let me reconsider. The total number of elements is at most 48.
    # Precompute for each x in 1..192:
    #   For each a in 1..192, a != x:
    #     For each b in 1..192, b != x, b > a:
    #       Compute (a+b) % x
    #   Store as a 2D array or use a smarter check.
    
    # Actually, let's precompute:
    # sum2_mod[x][a] = sorted list of (a+b) % x for all b != x, b > a
    # This is still large.
    
    # Better approach: Precompute a conflict matrix using bitsets.
    # For each x, define:
    #   triple_conflict_bits[x] = a list where for each a, we store a bitset of b's
    #   such that x | (a + b + c) for some c.
    
    # Actually, the cleanest precomputation:
    # For each x (1..192), for each a (1..192, a!=x), for each b (1..192, b!=x, b>a):
    #   remainder = (a+b) % x
    #   We need to know if there exists c in S\{x,a,b} such that c % x == (x - remainder) % x
    #   
    # So for each x, precompute: for each residue r in 0..x-1, a bitset of elements c (c!=x) with c%x==r
    # Then checking if x can be added: for each pair (a,b) in S, check if residue (x-(a+b)%x)%x has any element in S\{x,a,b}
    
    # Let's precompute: residue_bits[x][r] = bitset of c in 1..192, c!=x, with c%x==r
    # This is 192 * avg(192/2) ≈ 192*96 = ~18k bitsets, each 192 bits = 24 bytes. Total ~432KB. Fine.
    
    residue_bits = {}
    for x in range(1, n+1):
        residues = {}
        for c in range(1, n+1):
            if c == x:
                continue
            r = c % x
            if r not in residues:
                residues[r] = 0
            residues[r] |= (1 << (c-1))
        residue_bits[x] = residues
    
    # Now for checking if x can be added to set S (represented as a bitset or list):
    # Condition 1: x does not divide any triple sum in S
    #   For each pair (a,b) in S: need no c in S\{a,b} with c%x == (x - (a+b)%x) % x
    # Condition 2: For each a in S, a does not divide x+b+c for any b,c in S\{a}
    #   For each a in S: for each pair (b,c) in S\{a}: check if (x+b+c)%a != 0
    
    def can_add_bitset(x, S_bitset, S_list):
        """Check if x can be added to set S. S_bitset is int with bits set for elements in S."""
        m = len(S_list)
        if m < 3:
            # Check condition 2 only (condition 1 needs 3 elements from S)
            for a in S_list:
                others = [v for v in S_list if v != a]
                if len(others) >= 2:
                    for i in range(len(others)):
                        for j in range(i+1, len(others)):
                            if (x + others[i] + others[j]) % a == 0:
                                return False
            return True
        
        # Condition 1: x does not divide any triple sum in S
        x_residues = residue_bits[x]
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                target_r = (x - (a + b) % x) % x
                if target_r in x_residues:
                    # Check if any element c in S\{a,b} has residue target_r
                    c_bitset = x_residues[target_r]
                    # Remove a and b from consideration
                    mask = c_bitset & ~(1 << (a-1)) & ~(1 << (b-1))
                    # Check if any of these are in S
                    if mask & S_bitset:
                        return False
        
        # Condition 2: For each a in S, a does not divide x+b+c for any b,c in S\{a}
        for idx_a, a in enumerate(S_list):
            others = [v for v in S_list if v != a]
            if len(others) < 2:
                continue
            target = (-x) % a
            others_mod = [v % a for v in others]
            # Check all pairs
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    if (others_mod[i] + others_mod[j]) % a == target:
                        return False
        
        return True
    
    # Simulated annealing
    def is_valid_full(S_list):
        """Check if S_list is a valid set."""
        S_set = set(S_list)
        S_bitset = 0
        for v in S_list:
            S_bitset |= (1 << (v-1))
        for x in S_list:
            if not can_add_bitset(x, S_bitset & ~(1 << (x-1)), [v for v in S_list if v != x]):
                return False
        return True
    
    # Greedy initial solution
    def greedy_init():
        S = []
        S_bitset = 0
        S_set = set()
        # Try large numbers first
        for x in range(n, 0, -1):
            if time.time() > deadline - 5:
                break
            if can_add_bitset(x, S_bitset, S):
                S.append(x)
                S_bitset |= (1 << (x-1))
                S_set.add(x)
        return S, S_bitset
    
    # Local search / simulated annealing
    def local_search(S, S_bitset, max_iters=50000):
        """Try to add elements or swap to improve."""
        S_list = list(S)
        S_set = set(S_list)
        best = len(S_list)
        
        # First, try to add any element
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in range(1, n+1):
                if x in S_set:
                    continue
                if can_add_bitset(x, S_bitset, S_list):
                    S_list.append(x)
                    S_bitset |= (1 << (x-1))
                    S_set.add(x)
                    best = len(S_list)
                    improved = True
                    break
        
        # Then try swaps: remove one, add one
        if time.time() < deadline - 2:
            for _ in range(200):
                if time.time() > deadline - 2:
                    break
                if not S_list:
                    break
                # Try removing a random element and adding a different one
                idx = rng.randint(0, len(S_list)-1)
                removed = S_list.pop(idx)
                S_bitset &= ~(1 << (removed-1))
                S_set.discard(removed)
                
                added = False
                for x in range(1, n+1):
                    if x in S_set:
                        continue
                    if can_add_bitset(x, S_bitset, S_list):
                        S_list.append(x)
                        S_bitset |= (1 << (x-1))
                        S_set.add(x)
                        added = True
                        if len(S_list) > best:
                            best = len(S_list)
                        break
                if not added:
                    # Put it back
                    S_list.append(removed)
                    S_bitset |= (1 << (removed-1))
                    S_set.add(removed)
        
        return S_list, S_bitset, best
    
    # Multi-start with different orderings
    best_set = []
    best_size = 0
    
    # Start 1: greedy from large to small
    S, S_bitset = greedy_init()
    S, S_bitset, sz = local_search(S, S_bitset)
    if sz > best_size:
        best_size = sz
        best_set = list(S)
    
    if time.time() > deadline - 5:
        return best_set
    
    # Start 2: greedy from small to large (but prefer larger)
    S2 = []
    S2_bitset = 0
    S2_set = set()
    for x in range(1, n+1):
        if time.time() > deadline - 5:
            break
        if can_add_bitset(x, S2_bitset, S2):
            S2.append(x)
            S2_bitset |= (1 << (x-1))
            S2_set.add(x)
    S2, S2_bitset, sz2 = local_search(S2, S2_bitset)
    if sz2 > best_size:
        best_size = sz2
        best_set = list(S2)
    
    if time.time() > deadline - 5:
        return best_set
    
    # Start 3: randomized greedy
    for trial in range(10):
        if time.time() > deadline - 3:
            break
        S3 = []
        S3_bitset = 0
        S3_set = set()
        order = list(range(1, n+1))
        rng.shuffle(order)
        # Bias toward larger numbers
        order.sort(key=lambda x: -x + rng.random()*30)
        for x in order:
            if time.time() > deadline - 3:
                break
            if can_add_bitset(x, S3_bitset, S3):
                S3.append(x)
                S3_bitset |= (1 << (x-1))
                S3_set.add(x)
        S3, S3_bitset, sz3 = local_search(S3, S3_bitset)
        if sz3 > best_size:
            best_size = sz3
            best_set = list(S3)
    
    # Final improvement pass
    if best_set and time.time() < deadline - 1:
        S_final = list(best_set)
        S_final_set = set(S_final)
        S_final_bitset = 0
        for v in S_final:
            S_final_bitset |= (1 << (v-1))
        S_final, S_final_bitset, sz_final = local_search(S_final, S_final_bitset, max_iters=20000)
        if sz_final > best_size:
            best_size = sz_final
            best_set = list(S_final)
    
    return best_set
