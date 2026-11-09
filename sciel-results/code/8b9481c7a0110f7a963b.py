import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin for 150s limit
    
    # The problem is to minimize max_abs_autoconv / (sum(f))^2.
    # Let's analyze the scaling for a sequence of length N.
    # If f is random +/-1:
    #   sum(f) ~ sqrt(N) (in magnitude, roughly)
    #   (sum(f))^2 ~ N
    #   autoconv at lag k (k != 0) ~ sqrt(N) (random walk fluctuation)
    #   max_abs_autoconv ~ sqrt(N) * log(N) (extreme value of random walk)
    #   Score ~ sqrt(N)*log(N) / N = log(N) / sqrt(N)
    # This decreases as N increases.
    # For N=10^6, score ~ 13.8 / 1000 ~ 0.014.
    # For N=10^4, score ~ 9.2 / 100 ~ 0.09.
    # The seed program returns ones(N).
    #   sum(f) = N
    #   (sum(f))^2 = N^2
    #   autoconv at lag k = N - k
    #   max_abs_autoconv (excluding lag 0? usually yes, or lag 0 is N)
    #   If lag 0 is excluded, max is N-1.
    #   Score ~ N / N^2 = 1/N.
    # For N=10^6, score ~ 10^-6.
    # Wait, the seed score is 1.0. This implies the evaluator might be doing something specific.
    # Let's re-read the task carefully.
    # "score is the upper bound on C3' (maximum absolute autoconvolution)"
    # "evaluator measures the raw value (lower is better)"
    # "search score is seed/value, so 1.0 equals the seed program"
    # If seed is ones(N), value = 1/N (if lag 0 excluded) or 1 (if lag 0 included? No, 1/N is much smaller).
    # If the seed score is 1.0, and the formula is seed_value / value, then value must be equal to seed_value.
    # If seed_value is 1/N, then value must be 1/N.
    # If I return ones(N), I get 1.0.
    # If I return random +/-1, I get log(N)/sqrt(N).
    # For N=10^6, 1/N = 1e-6. log(N)/sqrt(N) ~ 0.014.
    # 1e-6 / 0.014 ~ 7e-5. This would be a terrible score (0.00007).
    # So random +/-1 is worse than ones(N) for large N?
    # Let's check N=1000.
    # ones: 1/1000 = 0.001.
    # random: log(1000)/sqrt(1000) ~ 6.9/31.6 ~ 0.22.
    # 0.001 / 0.22 ~ 0.0045. Still worse.
    # It seems ones(N) is very good because the denominator (sum)^2 scales as N^2, while numerator scales as N.
    # To beat ones(N), we need a sequence where max_abs_autoconv / (sum)^2 < 1/N.
    # i.e., max_abs_autoconv < N.
    # For ones(N), max_abs_autoconv (lag 1) is N-1 ~ N.
    # So we need max_abs_autoconv significantly smaller than N.
    # But sum(f) must be large.
    # If we use a sequence with many zeros, sum decreases.
    # If we use a sequence with mixed signs, sum decreases.
    # Consider a block of ones followed by a block of zeros.
    # f = [1...1 0...0] with K ones and N-K zeros.
    # sum = K. (sum)^2 = K^2.
    # autoconv is triangular. Peak at lag 0 is K. Lag 1 is K-1.
    # max_abs (lag != 0) is K-1.
    # Score = (K-1) / K^2 ~ 1/K.
    # If K < N, 1/K > 1/N. So this is worse than ones(N).
    
    # What if we use a sequence that cancels out in autoconv?
    # e.g. [1, -1, 1, -1 ...]
    # sum ~ 0. Denominator -> 0. Score -> inf. Bad.
    # We need large sum and small autoconv sidelobes.
    # This is the classic "low autocorrelation sequence" problem, but with a specific metric.
    # The metric is max_lag!=0 |R(lag)| / (sum f)^2.
    # Let S = sum f.
    # We want min max|R(lag)| / S^2.
    # For m-sequences (maximal length LFSR), periodic autocorrelation is -1/N (or similar).
    # Linear autocorrelation of m-sequence:
    # R(0) = N
    # R(k) for k != 0 is bounded by ~ sqrt(N) or N/2^m?
    # Actually, for m-sequence of length N=2^m-1:
    # Sum is 1 (one more +1 than -1). S^2 = 1.
    # Max autoconv sidelobe is ~ N/2^m ? No, it's bounded.
    # For m-sequence, linear autocorrelation sidelobes are bounded by 2^(m/2) roughly?
    # Let's check m=10, N=1023.
    # S=1. S^2=1.
    # Max sidelobe ~ 32?
    # Score ~ 32.
    # ones(1023) score ~ 1/1023 ~ 0.001.
    # 0.001 / 32 ~ 0.00003. Much worse.
    
    # It seems ones(N) is extremely hard to beat because the denominator is so large (N^2).
    # To get a score > 1.0, we need value < 1/N.
    # value = max_abs_autoconv / S^2.
    # So max_abs_autoconv / S^2 < 1/N.
    # max_abs_autoconv < S^2 / N.
    # Since S <= N (for +/-1), S^2/N <= N.
    # For ones(N), S=N, S^2/N = N. max_abs = N-1 < N. Holds.
    # If we reduce S, we reduce the RHS faster (quadratically) than the LHS (linearly? no, autoconv depends on structure).
    # If we flip one bit in ones(N):
    # S = N-2. S^2 ~ N^2.
    # Autoconv changes.
    # Flipping one bit from 1 to -1:
    # R(0) changes by -2.
    # R(k) changes by -2 * (number of matches).
    # The peak sidelobe might increase or decrease.
    # But S^2 decreases by ~4N.
    # So the ratio might increase.
    
    # Is there any sequence with score < 1/N?
    # Consider f = [1, 1, ..., 1] with some -1s carefully placed to cancel sidelobes while keeping sum large.
    # This is difficult.
    
    # Let's look at the inspirations.
    # Inspiration 2 suggests N=50,000.
    # Inspiration 3 uses m-sequences and local search. Score 0.0009 (terrible).
    # This confirms that m-sequences are bad for this specific metric because S is small.
    
    # So the strategy should be to keep S large (close to N) and reduce sidelobes.
    # Start with ones(N).
    # Flip bits to reduce max sidelobe.
    # Each flip reduces S by 2 (if 1->-1) or increases by 2 (if -1->1).
    # We want to keep S large. So we should mostly flip 1->-1? No, that reduces S.
    # Maybe we should start with a sequence that has S ~ N but better autocorrelation?
    # But ones(N) has the maximum possible S.
    # Any deviation reduces S.
    # If we flip k bits from 1 to -1:
    # S = N - 2k.
    # S^2 ~ N^2 - 4Nk.
    # The denominator decreases linearly with k (for small k).
    # The numerator (max sidelobe) for ones(N) is N-1.
    # Flipping bits can reduce the peak sidelobe.
    # Can we reduce the peak sidelobe from N-1 to something smaller, say N/2, by flipping O(N) bits?
    # If we flip N/4 bits randomly, S ~ N/2. S^2 ~ N^2/4.
    # Sidelobes of random +/-1 are ~ sqrt(N).
    # Score ~ sqrt(N) / (N^2/4) = 4 sqrt(N) / N^2 = 4 / N^1.5.
    # For N=10^6, 4 / 10^9 = 4e-9.
    # Seed score (ones) ~ 1e-6.
    # 1e-6 / 4e-9 = 250.
    # This is a HUGE improvement!
    
    # Wait, my previous analysis of random +/-1 was:
    # S ~ sqrt(N). S^2 ~ N.
    # Sidelobes ~ sqrt(N).
    # Score ~ sqrt(N) / N = 1/sqrt(N).
    # For N=10^6, 1/1000 = 0.001.
    # Seed score 1e-6.
    # 1e-6 / 0.001 = 0.001. Worse.
    
    # Why the discrepancy?
    # Case 1: Random +/-1. S ~ sqrt(N).
    # Case 2: Mostly ones, few flips. S ~ N.
    # Case 3: Half ones, half -1 (random). S ~ sqrt(N).
    
    # If we take a random +/-1 sequence, S is small.
    # If we take a sequence with bias, e.g. 75% ones, 25% -1.
    # S ~ 0.5 N. S^2 ~ 0.25 N^2.
    # Sidelobes?
    # The autoconvolution of a biased random sequence.
    # Mean value is 0.5.
    # The "DC" component of autoconv is S^2.
    # The sidelobes are fluctuations around the mean?
    # No, autoconv is sum f[i]f[i+k].
    # If f is i.i.d. with mean mu and variance sigma^2.
    # E[R(k)] = N mu^2.
    # Var[R(k)] ~ N sigma^4.
    # So R(k) ~ N mu^2 + sqrt(N) sigma^2.
    # We want max |R(k)| for k != 0.
    # If mu > 0, R(k) is positive and large (~ N mu^2).
    # The max will be close to N mu^2 + fluctuations.
    # Score = (N mu^2) / (N mu)^2 = 1/N.
    # Same as ones(N)!
    # The fluctuations are sqrt(N), which is small compared to N mu^2 (which is ~ N).
    # So the score is dominated by the mean, which is 1/N.
    # To beat 1/N, we need the mean to be smaller?
    # But the denominator is (N mu)^2 = N^2 mu^2.
    # Numerator is N mu^2.
    # Ratio is 1/N.
    # It seems for any sequence with non-zero mean, the score is ~ 1/N.
    # The only way to beat 1/N is if the autoconv sidelobes are significantly smaller than the mean.
    # i.e. we need cancellation in the sidelobes.
    # But the mean of R(k) is N mu^2.
    # If we can make R(k) close to 0 for k != 0, while keeping S large?
    # That's impossible if mu != 0, because R(k) ~ N mu^2.
    # Unless... the sequence is not stationary?
    # Or unless we use a sequence where the "DC" term cancels out in the sidelobes?
    # No, R(k) = sum f[i]f[i+k].
    # If f is constant, R(k) = N-k.
    # If f is random with mean 0, R(k) ~ 0. But S ~ 0.
    
    # So the key is: Can we have large S and small R(k) for k != 0?
    # This requires the sequence to be "self-cancelling" at non-zero lags, but "self-reinforcing" at lag 0?
    # Lag 0 is sum f[i]^2 = N (for +/-1).
    # We want R(k) << N for k != 0.
    # But S = sum f[i].
    # If S is large, the sequence has a strong DC component.
    # A strong DC component implies R(k) is large for all k.
    # Specifically, R(k) >= (S^2 - N)/2 ? No.
    # Parseval: sum R(k)^2 = sum |F(w)|^2.
    # R(0) = N.
    # If R(k) is small for k != 0, then sum R(k)^2 ~ N.
    # This is consistent with m-sequences (R(k) ~ 1).
    # But for m-sequences, S ~ 1.
    # Score ~ 1 / 1 = 1.
    # Seed score 1/N.
    # 1 / (1/N) = N. Worse.
    
    # It seems ones(N) is optimal or near-optimal for this metric.
    # Let's check if we can do slightly better than 1/N.
    # Score = max_{k!=0} |R(k)| / S^2.
    # For ones(N), R(1) = N-1. S^2 = N^2. Score = (N-1)/N^2 = 1/N - 1/N^2.
    # If we flip one bit at the end?
    # f = [1, ..., 1, -1].
    # S = N-2. S^2 = N^2 - 4N + 4.
    # R(1) = sum_{i=1}^{N-2} 1*1 + 1*(-1) = N-2 - 1 = N-3.
    # R(N-1) = 1*(-1) = -1.
    # Max abs is N-3.
    # Score = (N-3) / (N^2 - 4N + 4) ~ (N) / (N^2) = 1/N.
    # (N-3)/(N^2-4N) < (N-1)/N^2 ?
    # (N-3)N^2 < (N-1)(N^2-4N)
    # N^3 - 3N^2 < N^3 - 4N^2 - N^2 + 4N
    # N^3 - 3N^2 < N^3 - 5N^2 + 4N
    # 2N^2 < 4N -> N < 2. False for large N.
    # So flipping one bit makes it worse.
    
    # What if we flip many bits?
    # It seems hard to beat 1/N.
    # However, the task says "score is seed/value".
    # If I return ones(N), score is 1.0.
    # If I return a sequence with score 0.9 * (1/N), the search score is 1/0.9 = 1.11.
    # So I need to find a sequence with value < 1/N.
    
    # Is it possible?
    # We need max|R(k)| < S^2 / N.
    # For ones(N), max|R(k)| = N-1. S^2/N = N. N-1 < N. True.
    # The margin is small.
    # If we can reduce max|R(k)| while keeping S^2/N roughly constant, we win.
    # Reducing S reduces S^2/N.
    # So we need to reduce max|R(k)| faster than S^2.
    
    # Let's try a sequence with period 2: [1, -1, 1, -1 ...]
    # S ~ 0. Bad.
    # [1, 1, -1, -1, 1, 1, -1, -1 ...]
    # Period 4.
    # S ~ 0. Bad.
    
    # What if we use a sequence with a long run of ones, then a different pattern?
    # Maybe the seed program is not ones(N) in the evaluator?
    # "CURRENT PROGRAM (score 1.0000): return np.ones(N)"
    # So seed value is 1/N.
    
    # Let's try to construct a sequence with S ~ N and max|R(k)| ~ N/2.
    # This would give score ~ (N/2) / N^2 = 1/(2N).
    # Score improvement 2x.
    # How to get max|R(k)| ~ N/2?
    # The peak of ones(N) is at lag 1.
    # If we make the sequence "blocky", e.g. [1...1 0...0 1...1 0...0]?
    # No, zeros reduce S.
    
    # Maybe a random sequence with high bias?
    # p=0.9 for +1.
    # S ~ 0.8 N. S^2 ~ 0.64 N^2.
    # R(k) ~ N * 0.8^2 = 0.64 N.
    # Score ~ 0.64 N / 0.64 N^2 = 1/N.
    # Same.
    
    # It seems the score is robustly 1/N for any sequence with non-zero mean.
    # The only way to beat it is to have a sequence where the sidelobes are suppressed below the mean level.
    # This is the definition of a "good" autocorrelation sequence.
    # But good autocorrelation sequences (m-sequences, Gold codes) have small S.
    # There is a trade-off.
    # The metric penalizes small S heavily (quadratically in denominator).
    # So we want large S.
    # With large S, we have large mean autoconv.
    # We can only win if the fluctuations are negative?
    # No, max abs.
    
    # Let's look at the "Signed autocorrelation maximum absolute-value variant".
    # Maybe the evaluator includes lag 0?
    # If lag 0 is included:
    # R(0) = N.
    # Score = N / S^2.
    # For ones(N), S=N. Score = 1.
    # For random, S~sqrt(N). Score = N/N = 1.
    # For m-seq, S~1. Score = N.
    # If lag 0 is included, ones(N) gives 1.
    # Can we get < 1?
    # N / S^2 < 1 -> S^2 > N.
    # Max S is N. So S^2 = N^2 > N.
    # Score = N / N^2 = 1/N.
    # So if lag 0 is included, ones(N) gives 1/N.
    # And random gives 1.
    # So ones(N) is still best.
    
    # In both cases (lag 0 included or excluded), ones(N) seems to be the best or near-best.
    # However, the task asks to combine strategies.
    # Inspiration 3 uses local search.
    # Maybe local search on ones(N) can find a slightly better configuration?
    # Given the analysis, it's unlikely to improve significantly.
    # But I should implement a local search that tries to reduce the peak sidelobe while monitoring S.
    
    # Strategy:
    # 1. Start with ones(N).
    # 2. Compute score.
    # 3. Try flipping bits to reduce max sidelobe.
    # 4. Accept if score improves.
    # 5. Use time limit.
    
    # To make it faster, I'll use a smaller N for the search, then scale?
    # No, the score depends on N.
    # I'll use N=100,000 for the search to be safe with time.
    # And I'll try to find a sequence with score < 1/N.
    
    # One more idea:
    # What if we use a sequence that is 1 everywhere except a few -1s?
    # We analyzed 1 flip: worse.
    # What if we flip in a pattern that cancels the lag-1 peak?
    # Lag 1 peak is N-1.
    # To reduce it, we need f[i]f[i+1] to be -1 more often.
    # This means alternating signs.
    # But alternating signs reduces S.
    
    # I will implement a local search starting from ones(N).
    # I will also try starting from a random sequence with high bias (e.g. 80% ones).
    # And I will pick the best.

    N = 100_000
    
    def compute_score(f):
        n = len(f)
        s = np.sum(f)
        if abs(s) < 1e-10:
            return float('inf')
        nfft = 1
        while nfft < 2 * n:
            nfft <<= 1
        F = np.fft.rfft(f, n=nfft)
        A = np.fft.irfft(F * np.conj(F), n=nfft)[:n]
        # Exclude lag 0
        max_abs = np.max(np.abs(A[1:]))
        return max_abs / (s * s)
    
    def local_search(f, max_iters, time_limit):
        f = f.copy().astype(np.float32)
        n = len(f)
        
        def fast_autoconv(f):
            nfft = 1
            while nfft < 2 * n:
                nfft <<= 1
            F = np.fft.rfft(f, n=nfft)
            A = np.fft.irfft(F * np.conj(F), n=nfft)[:n]
            return A
        
        A = fast_autoconv(f)
        current_max = np.max(np.abs(A[1:]))
        current_sum = np.sum(f)
        current_score = current_max / (current_sum * current_sum) if abs(current_sum) > 1e-10 else float('inf')
        
        best_f = f.copy()
        best_score = current_score
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            i = np.random.randint(n)
            f[i] *= -1
            
            A_new = fast_autoconv(f)
            new_max = np.max(np.abs(A_new[1:]))
            new_sum = np.sum(f)
            if abs(new_sum) > 1e-10:
                new_score = new_max / (new_sum * new_sum)
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
                else:
                    f[i] *= -1
            else:
                f[i] *= -1
        
        return best_f, best_score

    # Candidate 1: Ones
    f_ones = np.ones(N, dtype=np.float32)
    score_ones = compute_score(f_ones)
    
    # Candidate 2: Local search from Ones
    f_ls_ones, score_ls_ones = local_search(f_ones, 1000, deadline - 10)
    
    # Candidate 3: High bias random
    np.random.seed(42)
    f_bias = np.where(np.random.rand(N) < 0.8, 1.0, -1.0).astype(np.float32)
    f_ls_bias, score_ls_bias = local_search(f_bias, 1000, deadline - 5)
    
    # Pick best
    candidates = [
        (f_ones, score_ones),
        (f_ls_ones, score_ls_ones),
        (f_ls_bias, score_ls_bias)
    ]
    
    best_f, best_score = min(candidates, key=lambda x: x[1])
    
    return best_f
