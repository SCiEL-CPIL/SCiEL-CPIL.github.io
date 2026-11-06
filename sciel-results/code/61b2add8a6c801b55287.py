import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin

    N = 16

    # Precompute all C(16,3) = 560 possible triples as bitmasks
    all_triple_masks = []
    all_triple_lists = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                mask = (1 << a) | (1 << b) | (1 << c)
                all_triple_masks.append(mask)
                all_triple_lists.append((a, b, c))
    
    num_possible_triples = len(all_triple_masks)
    
    # Precompute for each triple, the set of triples that conflict with it (share >= 2 points)
    # Two triples conflict if they share 2 or more points (since linear means at most 1 shared)
    # Actually, they conflict if they share a pair. Two triples share a pair iff their masks share >= 2 bits.
    # Let's precompute conflict masks or just check on the fly.
    # For a triple mask m, another triple mask n conflicts if (m & n) has bit_count >= 2.
    # Since masks have exactly 3 bits, m & n has 3, 2, or 1 bits if they share 3, 2, or 1 points.
    # They share 3 points only if m == n.
    # They share 2 points if popcount(m & n) == 2.
    # They share 1 point if popcount(m & n) == 1.
    # They share 0 points if popcount(m & n) == 0.
    # Conflict means popcount >= 2, which for distinct triples means popcount == 2.
    # So conflict if m != n and popcount(m & n) == 2.

    # Precompute: for each triple index i, a bitmask of indices j that conflict with i
    # 560 * 560 = 313,600 checks, each is a bitwise AND and popcount. Fast.
    conflict_masks = [0] * num_possible_triples
    for i in range(num_possible_triples):
        mi = all_triple_masks[i]
        for j in range(num_possible_triples):
            if i == j:
                continue
            mj = all_triple_masks[j]
            common = mi & mj
            # Check if common has exactly 2 bits set
            # popcount in Python 3.10+ is int.bit_count()
            if common.bit_count() == 2:
                conflict_masks[i] |= (1 << j)
    
    # Fast independence number calculation
    def check_alpha(triple_indices):
        """
        triple_indices: list of indices into all_triple_masks
        Returns the independence number.
        """
        if not triple_indices:
            return N
            
        # Build a lookup table: for each subset mask S (0 to 2^16-1), 
        # does S contain any triple?
        # Actually, we want to find the largest S that does NOT contain any triple.
        # Let's precompute a boolean array `contains` of size 2^16 = 65536.
        # contains[S] = True if S contains at least one triple from triple_indices.
        
        # We can compute this efficiently using DP:
        # contains[0] = False
        # For S > 0: contains[S] = contains[S without lowest bit] OR (S is exactly a triple)
        # Wait, that's not quite right. We need to check if ANY triple is a subset of S.
        # Standard subset sum / subset enumeration approach.
        
        # Alternative: For each triple mask m in triple_indices, mark all supersets of m as "contains".
        # Number of supersets of a 3-bit mask in 16 bits: 2^(16-3) = 8192.
        # 560 triples * 8192 = 4.5M operations. Too slow per restart if we do it from scratch.
        
        # Better: Precompute for the fixed set of all 560 triples? No, the set changes.
        
        # Let's use the DP approach:
        # contains[S] = any(contains[S without bit i] for i in S) OR (S in triple_set)
        # This is O(2^N * N) = 65536 * 16 = ~1M ops. Still too slow for 1000 restarts.
        
        # Let's try the superset marking but optimize it.
        # Actually, for N=16, 2^16 = 65536 is small.
        # Let's precompute `is_triple_mask` for the current set.
        # But the set changes.
        
        # Alternative fast check:
        # For each k from N down to 1:
        #   Check if there exists a subset of size k that is independent.
        #   A subset S is independent if no triple is contained in S.
        #   Equivalently, for every triple t, S does not contain t.
        #   We can check all subsets of size k. Max C(16,8) = 12870.
        #   For each subset, check if any triple is contained. 
        #   If we precompute a set of triple masks, we can check in O(1) if a given mask is a triple, 
        #   but we need to check if ANY triple is a subset, not just if S itself is a triple.
        
        # Let's stick with the DP approach but optimize it.
        # Actually, let's precompute `contains_any_triple` using the following:
        # Let T be the set of triple masks.
        # contains[S] = True if there exists t in T such that t is a subset of S.
        # We can compute this as:
        # contains[0] = False
        # For S from 1 to 2^16 - 1:
        #   lowbit = S & (-S)
        #   prev = S - lowbit
        #   contains[S] = contains[prev] or (S in T)
        #   Wait, this is wrong. S in T is not the only way.
        #   Actually, if prev contains a triple, then S contains it.
        #   But S might contain a triple that includes the lowbit.
        #   So we need: contains[S] = contains[prev] or (there exists t in T such that t includes lowbit and t is subset of S).
        #   This is getting complicated.
        
        # Let's just use the brute force over subsets for now, since N=16 is small.
        # But 1000 restarts * 12870 subsets * 560 triples = 7.2 billion ops. Too slow.
        
        # Let's use the superset marking approach but only for the current set of triples.
        # For each triple in the current set, mark all its supersets.
        # Number of triples in current set is at most 560.
        # Each marking is 2^13 = 8192 operations.
        # 560 * 8192 = 4.5M ops per restart. 1000 restarts = 4.5B ops. Still too slow.
        
        # We need a faster method.
        
        # Alternative: Use inclusion-exclusion or fast zeta transform.
        # Let f[S] = 1 if S is a triple, 0 otherwise.
        # We want g[S] = sum_{t subset S} f[t]. Then g[S] > 0 means S contains a triple.
        # We can compute g using the fast zeta transform in O(N * 2^N) = 16 * 65536 = 1M ops.
        # This is fast enough!
        
        # Precompute the zeta transform for the current set of triples.
        # But the set changes. So we need to recompute it for each restart.
        # 1M ops per restart * 1000 restarts = 1B ops. Might be too slow in Python.
        
        # Let's try a different approach:
        # Since N=16, we can precompute for each mask S, the list of triples contained in S.
        # But that's 65536 * (average number of triples in S). Too much memory.
        
        # Let's go back to the brute force over subsets of size k, but optimize the inner check.
        # For a given subset S, we want to check if any triple is contained in S.
        # We can precompute for each pair of points, the list of triples containing that pair.
        # Then for a subset S, we can check if there's a triple contained in S by:
        #   For each pair (a,b) in S:
        #       For each triple (a,b,c) containing that pair:
        #           If c is in S: then S contains this triple.
        # Number of pairs in S is C(k,2). For k=8, that's 28 pairs.
        # Each pair is in C(16-2, 1) = 14 triples? No, C(14,1)=14? No, C(16-2, 3-2)=C(14,1)=14.
        # So 28 * 14 = 392 checks per subset.
        # 12870 subsets * 392 = 5M ops per k.
        # For k from 16 down to 1, worst case 16 * 5M = 80M ops. Too slow.
        
        # Let's use the zeta transform approach but implement it in C speed using numpy?
        # No, we can't use numpy for this because the set changes.
        
        # Actually, let's precompute the zeta transform once for all 560 triples, and then for a subset of triples, 
        # we can't easily update it.
        
        # Let's try a different angle: 
        # The independence number is the largest k such that there exists a set of k points with no triple.
        # This is equivalent to: the minimum number of points to hit all triples is N - alpha.
        # This is the hitting set problem, which is NP-hard, but for N=16 we can use bitmask DP.
        
        # DP over subsets:
        # Let dp[S] = True if S is a hitting set for all triples.
        # dp[0] = (no triples)
        # dp[S] = dp[S without lowbit] AND (S without lowbit is not a triple? No.)
        # Actually, S is a hitting set if for every triple t, S intersects t.
        # dp[S] = all(S & t != 0 for all t in triples)
        # We can compute this using:
        # dp[S] = dp[prev] AND (S & t_lowbit != 0) ??? No.
        
        # Let's just use the brute force with early termination.
        # For k from N down to 1:
        #   For each subset S of size k:
        #       Check if S is independent (no triple contained in S).
        #       If yes, return k.
        # The check for independence:
        #   For each triple t:
        #       If t is a subset of S: return False
        #   Return True
        # We can optimize by precomputing for each triple t, a bitmask t_mask.
        # Then t is subset of S iff (t_mask & S_mask) == t_mask.
        # Number of triples is at most 560.
        # Number of subsets of size k is C(16,k).
        # For k=8, 12870 * 560 = 7.2M ops.
        # For k=9, 11440 * 560 = 6.4M ops.
        # Total for k=16 down to 1: sum C(16,k) * 560 = 65535 * 560 = 36.7M ops.
        # This is borderline in Python. Might take 1-2 seconds per restart.
        # 1000 restarts = 1000-2000 seconds. Too slow.
        
        # We need a faster check_alpha.
        
        # Let's use the following insight:
        # We can precompute for each mask S (0 to 2^16-1), whether S contains any triple from the full set of 560 triples.
        # But the set of triples in the current solution is a subset of the 560.
        # So we can't precompute it once.
        
        # Alternative: Use random sampling to estimate alpha, but we need exact value.
        
        # Let's try to use the conflict structure.
        # If two triples share 2 points, they conflict.
        # The independence number is related to the structure of the hypergraph.
        
        # Given the time constraints, let's use a hybrid approach:
        # 1. For small number of triples (< 100), use brute force over subsets.
        # 2. For larger number, use a heuristic or approximation.
        
        # But we need exact value for scoring.
        
        # Let's optimize the brute force:
        # Precompute for each point, the list of triples containing it.
        # Then for a subset S, to check if it contains a triple:
        #   For each point a in S:
        #       For each triple (a,b,c) containing a:
        #           If b in S and c in S: return False
        # Number of triples containing a point is C(15,2) = 105.
        # For |S|=k, we check k * 105 triples.
        # For k=8, 8*105 = 840 checks.
        # 12870 * 840 = 10.8M ops.
        # Total 65535 * 840 = 55M ops. Still too slow.
        
        # Let's just accept that check_alpha is slow and reduce the number of restarts.
        # Or, let's use a faster method:
        # Precompute a table `has_triple[S]` for the current set of triples using the zeta transform.
        # Implement zeta transform in Python.
        
        # Zeta transform:
        # f = [0] * (1 << N)
        # for t in triple_masks: f[t] = 1
        # for i in range(N):
        #     for mask in range(1 << N):
        #         if mask & (1 << i):
        #             f[mask] |= f[mask ^ (1 << i)]
        # Now f[S] = 1 iff S contains a triple.
        
        # This is O(N * 2^N) = 16 * 65536 = 1M ops.
        # In Python, this might take 0.1-0.5 seconds.
        # 1000 restarts = 100-500 seconds. Might be too slow.
        
        # Let's implement it and see.
        
        f = [0] * (1 << N)
        for idx in triple_indices:
            f[all_triple_masks[idx]] = 1
        
        for i in range(N):
            bit = 1 << i
            for mask in range(1 << N):
                if mask & bit:
                    f[mask] |= f[mask ^ bit]
        
        # Now find the largest k such that there exists a subset of size k with f[S] = 0.
        # Iterate k from N down to 1:
        for k in range(N, 0, -1):
            found = False
            for subset in combinations(range(N), k):
                mask_s = 0
                for p in subset:
                    mask_s |= (1 << p)
                if f[mask_s] == 0:
                    found = True
                    break
            if found:
                return k
        return 0

    def is_linear(triples):
        used_pairs = set()
        for t in triples:
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            for p in pairs:
                if p in used_pairs or (p[1], p[0]) in used_pairs:
                    return False
                used_pairs.add(p)
        return True

    best_triples = None
    best_alpha = N + 1
    
    num_restarts = 1000
    
    for i in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
            
        random.seed(i * 101 + 7)
        
        # Greedy construction
        triples = []
        used_pairs = set()
        
        # Shuffle all possible triples
        all_triples_idx = list(range(num_possible_triples))
        random.shuffle(all_triples_idx)
        
        for idx in all_triples_idx:
            a, b, c = all_triple_lists[idx]
            pairs = [(a, b), (a, c), (b, c)]
            conflict = False
            for p in pairs:
                if p in used_pairs or (p[1], p[0]) in used_pairs:
                    conflict = True
                    break
            if not conflict:
                triples.append([a, b, c])
                for p in pairs:
                    used_pairs.add(p)
        
        if not triples:
            continue
            
        # Convert to indices
        triple_indices = []
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            # Find index
            # This is O(num_possible_triples) which is 560.
            # We can precompute a dict from mask to index.
            pass
        
        # Precompute mask to index dict
        # Let's do it once outside the loop.
        
        if i == 0:
            mask_to_idx = {}
            for idx in range(num_possible_triples):
                mask_to_idx[all_triple_masks[idx]] = idx
        
        triple_indices = []
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_indices.append(mask_to_idx[mask])
            
        # Calculate alpha
        alpha = check_alpha(triple_indices)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            
        # If we found the theoretical minimum (5), we can stop early?
        if best_alpha <= 5:
            break

    if best_triples is None:
        # Fallback: return a valid linear system (even if alpha is high)
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]
        
    return best_triples
