"""Rewritten program: longest prime subsequence with non-decreasing gaps via DP."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes <= limit using a sieve."""
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s kill

    primes = _sieve_primes(10000)
    n = len(primes)
    
    # For each prime index i, we want to find the longest valid subsequence ending at i.
    # A valid subsequence has non-decreasing gaps.
    # Let dp[i] = max length of valid subsequence ending at prime i.
    # Let prev[i] = index of the previous prime in the optimal subsequence ending at i (for reconstruction).
    # Let last_gap[i] = the gap used in the optimal subsequence ending at i (i.e., primes[i] - primes[prev[i]]).
    
    # However, the standard DP "dp[i] = max over j < i where gap(j,i) >= last_gap[j] of dp[j] + 1" 
    # is not quite right because we need to track the last gap.
    # Actually, if we define the state as (last_prime_index, last_gap), it's too large.
    
    # Let's re-evaluate. The constraint is on the gaps between consecutive primes in the subsequence.
    # Let the subsequence be p_{i1}, p_{i2}, ..., p_{ik} with i1 < i2 < ... < ik.
    # Gaps are g_m = p_{i(m+1)} - p_{im} for m=1..k-1.
    # We require g_1 <= g_2 <= ... <= g_{k-1}.
    
    # This is a longest path problem in a DAG. Nodes are primes.
    # Edge from j to i (j < i) exists if we can follow a path ending at j with last gap <= (p_i - p_j).
    # But the "last gap" depends on the specific path, not just the node j.
    
    # So we need a different DP state.
    # Let f(i, g) be the length of the longest valid subsequence ending at prime i with last gap exactly g?
    # No, last gap can be any value.
    
    # Alternative:
    # Let dp[i] be the maximum length of a valid subsequence ending at prime i.
    # This is not sufficient because the validity of extending the subsequence depends on the *last* gap.
    # Example: Subsequence A ends at i with last gap 10. Subsequence B ends at i with last gap 5.
    # If we extend to j where p_j - p_i = 7, A cannot be extended (7 < 10), but B can (7 >= 5).
    # So we need to know the last gap.
    
    # Let's define:
    # best_len[i][g] = max length of valid subsequence ending at prime i with last gap = g.
    # But g can be up to 10000, and i up to 1229. 1229 * 10000 is ~12 million states.
    # Transitions: for each state (i, g), we can extend to j > i if p_j - p_i >= g.
    # This seems heavy.
    
    # Let's try a different perspective.
    # Since the gaps are non-decreasing, the gaps in the subsequence are a non-decreasing sequence.
    # Let the gaps be d_1, d_2, ..., d_{k-1}.
    # d_1 <= d_2 <= ... <= d_{k-1}.
    # The primes are p_1, p_1+d_1, p_1+d_1+d_2, ...
    
    # Maybe we can iterate over possible last gaps?
    # Or iterate over the first prime?
    
    # Let's try the DP with state (i, last_gap_index) where last_gap_index refers to the gap value.
    # Actually, the gap values are differences between primes, so there are at most O(n^2) possible gaps, but many are unique.
    # However, for a fixed i, the possible last gaps are p_i - p_j for j < i. There are i-1 such gaps.
    # So total states are sum_{i=1}^n (i-1) = O(n^2). n=1229, so ~750k states. This is manageable.
    
    # Let's define:
    # dp[i][j] = length of longest valid subsequence ending at prime i, where the previous prime in the subsequence is prime j (j < i).
    # The last gap is p_i - p_j.
    # To transition from (i, j) to (k, i) where k > i, we need p_k - p_i >= p_i - p_j, i.e., p_k - p_i >= p_i - p_j.
    # dp[k][i] = max(dp[k][i], dp[i][j] + 1) if p_k - p_i >= p_i - p_j.
    
    # Base case: dp[i][j] = 2 for any j < i (subsequence of length 2: p_j, p_i).
    # We want max over all i, j of dp[i][j].
    
    # This is O(n^3) if done naively. n=1229, n^3 ~ 1.8e9. Too slow.
    
    # Optimization:
    # For a fixed k, we want to find max over i < k such that p_k - p_i >= p_i - p_j, i.e., 2*p_i <= p_k + p_j, i.e., p_i <= (p_k + p_j)/2.
    # Wait, the condition is p_k - p_i >= p_i - p_j => p_k + p_j >= 2*p_i => p_i <= (p_k + p_j)/2.
    # So for fixed k and j, we need max dp[i][j] over i < k with p_i <= (p_k + p_j)/2.
    # This is still complex.
    
    # Let's try a different approach: iterate over the last gap.
    # Or, use the fact that we only need the max length, not the exact subsequence (though we need to return it).
    
    # Let's try a greedy or heuristic approach first to see if we can beat the baseline.
    # The baseline is a greedy that picks primes with non-decreasing gaps.
    
    # Let's try to improve the greedy by looking ahead or using multiple starting points.
    
    # Actually, let's try the O(n^3) DP but with numpy optimization.
    # n = 1229. n^3 = 1.86e9. Even with numpy, this might be too slow.
    
    # Let's try to reduce the complexity.
    # For each i, the possible previous primes j are those with p_i - p_j being a valid last gap.
    # But the validity depends on the previous gap.
    
    # Let's try a different DP formulation.
    # Let f(i, g) = max length of valid subsequence ending at prime i with last gap <= g.
    # Then f(i, g) = max(f(i, g-1), max_{j: p_i - p_j == g} (1 + max_{h: p_j - p_h <= g} f(j, p_j - p_h)))
    # This seems complicated.
    
    # Let's try a simple improvement to the greedy:
    # Try all possible starting primes and greedily extend, keeping the best.
    
    best_seq = []
    for start_idx in range(n):
        if time.time() > deadline:
            break
        seq = [primes[start_idx]]
        last_gap = 0
        for next_idx in range(start_idx + 1, n):
            gap = primes[next_idx] - primes[start_idx + len(seq) - 1]
            # Wait, the next prime must be after the last prime in the sequence.
            # The greedy approach in the baseline just iterates through primes in order.
            # Let's try a different greedy: from the last prime in the sequence, pick the smallest next prime that gives a gap >= last_gap.
            pass
    
    # Let's implement the "smallest next prime with gap >= last_gap" greedy for each starting prime.
    best_seq = []
    for start_idx in range(n):
        if time.time() > deadline:
            break
        seq = [primes[start_idx]]
        last_gap = 0
        last_prime = primes[start_idx]
        # Find the next prime
        next_idx = start_idx + 1
        while next_idx < n:
            gap = primes[next_idx] - last_prime
            if gap >= last_gap:
                seq.append(primes[next_idx])
                last_gap = gap
                last_prime = primes[next_idx]
                next_idx += 1
            else:
                # We need a larger gap, so we must skip to a larger prime.
                # But the gap increases as we go to larger primes.
                # So we can binary search for the smallest prime p such that p - last_prime >= last_gap.
                # p >= last_prime + last_gap.
                target = last_prime + last_gap
                # Find the first prime >= target.
                # Use bisect.
                import bisect
                pos = bisect.bisect_left(primes, target, next_idx)
                if pos < n:
                    gap = primes[pos] - last_prime
                    seq.append(primes[pos])
                    last_gap = gap
                    last_prime = primes[pos]
                    next_idx = pos + 1
                else:
                    break
        if len(seq) > len(best_seq):
            best_seq = seq
    
    return best_seq
# EVOLVE-BLOCK-END
