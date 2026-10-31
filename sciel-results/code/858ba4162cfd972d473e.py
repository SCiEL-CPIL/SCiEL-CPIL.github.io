"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _max_sidelobe(f):
    """Compute max |autocorrelation| for lags 1..N-1, normalized by N."""
    n = len(f)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    
    F = np.fft.rfft(f, nfft)
    acf = np.fft.irfft(F * np.conj(F), nfft)[:n]
    # Exclude lag 0, take max absolute value of remaining lags
    if n > 1:
        return np.max(np.abs(acf[1:])) / n
    return 0.0


def _soft_max_sidelobe(f, beta=50.0):
    """
    Smooth approximation of max |sidelobes| / N using log-sum-exp.
    This is differentiable and suitable for gradient-based optimization.
    """
    n = len(f)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    
    F = np.fft.rfft(f, nfft)
    acf = np.fft.irfft(F * np.conj(F), nfft)[:n]
    sidelobes = acf[1:]
    
    # We want max(|x_i|) / n
    # max(|x_i|) = max(max(x_i), max(-x_i))
    # Approximate max(x_i) via log-sum-exp
    # To avoid overflow, subtract max before exp
    x = sidelobes
    mx = np.max(x)
    mx_neg = np.max(-x)
    
    # log-sum-exp(x) = mx + log(sum(exp(x - mx)))
    # log-sum-exp(-x) = mx_neg + log(sum(exp(-x - mx_neg)))
    
    # If mx or mx_neg is very large, exp might overflow if not careful, 
    # but since we subtract max, the exponent is <= 0.
    
    # sum_exp_pos = sum(exp(x - mx))
    # sum_exp_neg = sum(exp(-x - mx_neg))
    
    # log_max_pos = mx + log(sum_exp_pos)
    # log_max_neg = mx_neg + log(sum_exp_neg)
    
    # max(|x|) approx exp( (log_max_pos + log_max_neg) / 2 ? No.
    # max(|x|) = max( exp(log_max_pos), exp(log_max_neg) )
    # = exp( max( log_max_pos, log_max_neg ) )
    
    # Let L_pos = mx + log(sum(exp(x - mx)))
    # Let L_neg = mx_neg + log(sum(exp(-x - mx_neg)))
    
    # L = max(L_pos, L_neg)
    # Result = exp(L) / n
    
    # However, for beta-softened max:
    # max_beta(x) = (1/beta) * log(sum(exp(beta * x)))
    # max_beta(|x|) = max( max_beta(x), max_beta(-x) )
    
    # Let's just use L_p norm for simplicity and stability, p=10 or 20.
    # It's a good proxy for max and differentiable.
    p = 10
    val = (np.sum(np.abs(sidelobes) ** p)) ** (1.0 / p)
    return val / n


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Initialize with random signs
    best_f = rng.choice([-1.0, 1.0], size=N)
    best_score = _max_sidelobe(best_f)
    
    # Phase 1: Greedy Local Search with Random Restarts
    # This handles the discrete landscape well.
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Start from best found so far or a new random initialization
        if restart == 0:
            current_f = best_f.copy()
            current_score = best_score
        else:
            # Occasionally start from a structured signal?
            # For now, stick to random signs for diversity
            current_f = rng.choice([-1.0, 1.0], size=N)
            current_score = _max_sidelobe(current_f)
            
            if current_score < best_score:
                best_score = current_score
                best_f = current_f.copy()
        
        # Greedy local search: try flipping each position in random order
        # Use a smaller number of passes per restart to allow more restarts
        improved = True
        # Limit passes per restart to avoid getting stuck in one local min for too long
        max_passes = 10
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            if not improved:
                break
            improved = False
            order = rng.permutation(N)
            for i in order:
                if time.time() > deadline:
                    break
                
                # Flip sign at position i
                current_f[i] *= -1.0
                new_score = _max_sidelobe(current_f)
                
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                    if current_score < best_score - 1e-12:
                        best_score = current_score
                        best_f = current_f.copy()
                else:
                    # Revert the flip
                    current_f[i] *= -1.0
        
        # If we hit the deadline, break out
        if time.time() > deadline:
            break
            
    # Phase 2: Spectral Refinement (Structural Change)
    # The directive suggests modeling as a smooth window.
    # We can use the best binary sequence as an initial guess for a continuous optimization,
    # or just optimize a parametric window directly.
    # Given the "signed step function" constraint, the values must likely be +/- 1.
    # However, the inspirations show that real-valued windows score better (0.05 vs 0.012).
    # The task says "Choose a signed step function". This usually implies values in {-1, 1}.
    # BUT, inspiration 2 (score 0.389) returned a Kaiser window (real values).
    # And the cell finding says: "Switching from binary sequences to real-valued window functions... Measured: 0.0007547 -> 0.05413."
    # Wait, the score 0.05413 is HIGHER (worse) than 0.0125?
    # "score is the upper bound on C3". "lower is better".
    # Seed score 1.0.
    # Current program score 0.0125.
    # Inspiration 2 score 0.3890.
    # Cell finding says "improved island 6" with score 0.05413.
    # Let's re-read: "Measured: 0.0007547 -> 0.05413". This looks like a raw value change?
    # If raw value dropped from 0.00075 to 0.054, that's worse.
    # If score improved, raw value must drop.
    # The cell finding says "Real-valued windows can achieve sidelobes orders of magnitude lower... likely jumping the score from ~0.0008 to <0.0001."
    # This implies the RAW VALUE dropped.
    # But Inspiration 2 (Kaiser) had a SCORE of 0.3890.
    # The current program has a SCORE of 0.0125.
    # Lower score is better? "search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement".
    # Wait. "score is the upper bound on C3". "The evaluator measures the raw value (lower is better)."
    # "search score is seed/value".
    # If raw value is low, search score is high.
    # Current program: Score 0.0125. This means seed/value = 0.0125.
    # This implies value = seed / 0.0125.
    # If seed is 1.0, value is ~80.
    # Inspiration 2: Score 0.3890. Value = 1/0.389 ~ 2.57.
    # So Inspiration 2 is BETTER than Current Program?
    # But Current Program is the "CURRENT PROGRAM (score 0.0125)".
    # Usually in these tasks, the "current program" is the one being evolved, and it might be the best in the lineage so far, or just the parent.
    # The prompt says "CURRENT PROGRAM (score 0.0125)".
    # And "INSPIRATIONS... inspiration 2 (score 0.3890)".
    # If 0.3890 > 0.0125, and higher is better (improvement), then Inspiration 2 is better.
    # Let's check the cell finding again: "Measured: 0.0007547 -> 0.05413".
    # If these are raw values, 0.054 > 0.00075, so it got worse.
    # But it says "improved island 6".
    # Maybe the scores in the header are different metrics?
    # "score is the upper bound on C3".
    # "search score is seed/value".
    # If the header "score 0.0125" refers to the SEARCH SCORE, then Value = 1/0.0125 = 80.
    # If Inspiration 2 has Search Score 0.3890, Value = 1/0.389 = 2.57.
    # 2.57 < 80, so Inspiration 2 is much better.
    # Why did the cell finding say 0.00075 -> 0.054?
    # Maybe those were intermediate raw values during optimization?
    # Or maybe I am misinterpreting the "score" in the header.
    # Let's look at Inspiration 1: `return np.ones(N)`. Score 1.0000.
    # If f = ones(N), autocorrelation is N, N-1, ..., 1.
    # Max sidelobe is N-1. Normalized by N: (N-1)/N ~ 1.
    # Raw value ~ 1.
    # Search score = seed / value = 1 / 1 = 1.
    # So for Inspiration 1, Raw Value = 1, Search Score = 1.
    
    # Now Current Program: Search Score 0.0125.
    # Raw Value = 1 / 0.0125 = 80.
    # This is WORSE than 1.
    # How can a random sign sequence have a max sidelobe of 80 (normalized by N=400, so 80/400 = 0.2 of N)?
    # For random signs, autocorrelation sidelobes are typically ~ sqrt(N).
    # sqrt(400) = 20.
    # Normalized by N: 20/400 = 0.05.
    # Raw value 0.05.
    # Search score = 1 / 0.05 = 20.
    # So a random sign sequence should have a score around 20.
    # The current program has a score of 0.0125, which implies a raw value of 80.
    # This is extremely high. It suggests the current program is returning something very bad, or the metric is inverted in my head.
    
    # Let's re-read: "score is the upper bound on C3".
    # "The evaluator measures the raw value (lower is better)."
    # "The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement".
    
    # If Seed = 1 (from Inspiration 1), and Value = 1, Score = 1.
    # If Current Program Score = 0.0125, then Value = 1/0.0125 = 80.
    # This means the current program is producing a signal with a max sidelobe of 80 (normalized).
    # This is worse than the seed.
    
    # Why?
    # Maybe the "seed" is not 1.0?
    # "1.0 equals the seed program".
    # Inspiration 1 is `np.ones(N)`.
    # If `np.ones(N)` is the seed, then Seed Value = 1.
    # Then Current Program Value = 80.
    # This makes no sense for a local search on random signs, which should find ~0.05.
    
    # Alternative interpretation:
    # Maybe the "score" in the header IS the raw value?
    # "CURRENT PROGRAM (score 0.0125)".
    # If Raw Value = 0.0125.
    # Then Search Score = 1 / 0.0125 = 80.
    # This would be an improvement over Seed (Score 1).
    # And Inspiration 2 (score 0.3890) would have Raw Value 0.389.
    # 0.389 > 0.0125, so Current Program is better than Inspiration 2.
    # This fits the "lower is better" for raw value.
    # And 0.0125 is a very good raw value for N=400 (ideal is 0, random is ~0.05).
    # So it is highly likely that the "score" listed in the header is the RAW VALUE, not the search score.
    # OR, the "score" listed is the search score, and the current program is bad.
    # But the prompt says "CURRENT PROGRAM (score 0.0125)" and "INSPIRATIONS... (score 1.0000)".
    # Usually, the current program is the one being evolved, and it's usually decent.
    # If the current program had a search score of 0.0125, it would be terrible.
    # If the current program has a raw value of 0.0125, it is excellent.
    
    # Let's assume the header "score" is the RAW VALUE for the purpose of "lower is better".
    # But the text says: "The search score is seed/value... 1.0 equals the seed... above 1.0 is an improvement".
    # And "The evaluator measures the raw value".
    # The field name is "score".
    # In many of these SCiEL tasks, the "score" displayed is the metric being optimized.
    # If the metric is "upper bound on C3", and lower is better, then the score should be lower.
    # Let's assume the numbers in the header are the RAW VALUES.
    # Current: 0.0125 (Good)
    # Insp 1: 1.0 (Bad, seed)
    # Insp 2: 0.389 (Worse than current)
    
    # If this assumption is correct, the current program is already quite good.
    # The goal is to improve it further (lower the raw value).
    # The strategy from Inspiration 2 (Kaiser window) gave 0.389, which is worse.
    # Why? Because Kaiser windows are not signed step functions. They are continuous.
    # The task requires a "signed step function".
    # If the evaluator enforces `f[i] in {-1, 1}`, then returning a Kaiser window might be invalid or clamped.
    # If it's clamped to +/-1, it becomes a random-ish sign sequence, which is ~0.05.
    # But Insp 2 got 0.389. This is worse than random.
    # Maybe the Kaiser window has high sidelobes in the discrete autocorrelation sense?
    # Or maybe the "score" in the header IS the search score?
    # If Current Score (Search) = 0.0125, it's bad.
    # If Insp 2 Score (Search) = 0.389, it's better.
    # If Insp 1 Score (Search) = 1.0, it's the seed.
    # This ordering (0.0125 < 0.389 < 1.0) makes sense if higher is better.
    # So:
    # Seed (Ones): Search Score 1.0.
    # Insp 2 (Kaiser): Search Score 0.389.
    # Current (Binary Search): Search Score 0.0125.
    
    # This implies the Current Program is the WORST of the three.
    # Why would a binary local search be so bad?
    # Maybe it's getting stuck in a bad local minimum?
    # Or maybe the "score" in the header is indeed the raw value, and the current program is the BEST.
    # Let's look at the cell finding: "Measured: 0.0007547 -> 0.05413".
    # If these are raw values, 0.00075 is very good. 0.054 is worse.
    # It says "improved island 6".
    # This is confusing.
    
    # Let's try a different angle.
    # The task is "Third autocorrelation inequality".
    # C3 is likely a constant related to the 3rd moment or something.
    # The "upper bound on C3" is what we want to minimize.
    # The raw value is this upper bound.
    # Lower raw value is better.
    # Search Score = Seed / Raw Value.
    # Higher Search Score is better.
    
    # If Current Program Raw Value = 0.0125.
    # Search Score = 1 / 0.0125 = 80.
    # If Insp 2 Raw Value = 0.389.
    # Search Score = 1 / 0.389 = 2.57.
    # If Insp 1 Raw Value = 1.0.
    # Search Score = 1 / 1.0 = 1.
    
    # In this case, Current Program (80) > Insp 2 (2.57) > Insp 1 (1).
    # This makes the Current Program the best.
    # This aligns with "CURRENT PROGRAM (score 0.0125)" being the one we are evolving, and it being good.
    # The "score" in the header is the RAW VALUE.
    # The "search score" mentioned in the text is a different metric (inverse).
    # The evaluator measures the raw value.
    # So I should aim to LOWER the raw value.
    
    # Current Raw Value: 0.0125.
    # This is quite low. Random is ~0.05.
    # Can we go lower?
    # The theoretical lower bound for binary sequences is related to 1/sqrt(N) or similar?
    # Actually, for binary sequences, the max sidelobe can be as low as ~sqrt(N) in absolute terms.
    # Normalized by N, that's 1/sqrt(N).
    # For N=400, 1/20 = 0.05.
    # Wait, if random is ~sqrt(N), then normalized is ~1/sqrt(N).
    # 1/sqrt(400) = 0.05.
    # But the current program achieved 0.0125.
    # This is 4x better than random.
    # This suggests the local search found a very good sequence.
    
    # Can we do better?
    # The directive suggests a STRUCTURAL change: "Model the signed step function as a discretized finite impulse response (FIR) filter with a symmetric amplitude spectrum... smooth, band-limited window".
    # But the signal must be a "signed step function".
    # If I use a smooth window, it's not a signed step function.
    # Unless I quantize it to +/-1.
    # If I quantize a smooth window to +/-1, I get a binary sequence.
    # The structure of the window (e.g., Kaiser, Gaussian) might provide a better starting point or constraint for the binary search.
    # For example, a Kaiser window has a smooth transition. A binary sequence derived from it would have long runs of 1s and -1s.
    # Random signs have short runs.
    # Maybe sequences with specific run-length properties have lower autocorrelation sidelobes?
    
    # Strategy:
    # 1. Generate a parametric smooth window (e.g., Kaiser, or a cosine-tapered window).
    # 2. Quantize it to +/-1.
    # 3. Use this as the initial guess for the binary local search.
    # 4. Alternatively, optimize the parameters of the window directly, but keep the output binary.
    #    This is a mixed discrete-continuous problem.
    #    We can optimize the continuous parameters (like beta for Kaiser) to minimize the autocorrelation of the QUANTIZED signal.
    
    # Let's try optimizing the Kaiser beta parameter.
    # For each beta, generate Kaiser window, quantize to +/-1, compute raw value.
    # Find the beta that minimizes the raw value.
    # Then run local search on that best quantized sequence.
    
    # This combines the "structural idea" of Inspiration 2 (Kaiser window) with the "refinement machinery" of the Current Program (local search).
    
    # Implementation:
    # 1. Loop over a range of Kaiser betas.
    # 2. For each beta, w = kaiser(N, beta).
    # 3. f = sign(w) (or map to -1, 1). Note: kaiser is positive, so sign is always 1?
    #    Ah, Kaiser window is positive. sign(w) is always 1. That's just np.ones(N).
    #    We need a signed window.
    #    Maybe use a window that goes negative?
    #    Or use a modulated window?
    #    Or simply: f[i] = 1 if w[i] > threshold else -1.
    #    If we use a symmetric window, the center is high, edges are low.
    #    Threshold at 0? No, Kaiser is > 0.
    #    Threshold at mean?
    #    Or use a window that oscillates?
    #    The inspiration 2 used `np.kaiser(N, beta)` and returned it.
    #    But that returned real values.
    #    If the evaluator accepts real values, then we don't need to quantize.
    #    But the task says "signed step function".
    #    If the evaluator enforces binary, then real values are clamped or rejected.
    #    If Insp 2 got a score of 0.389 (raw value), and it returned real values, maybe the evaluator clamped them?
    #    If clamped to +/-1, a Kaiser window (positive) becomes all 1s.
    #    All 1s has raw value ~1.
    #    But Insp 2 got 0.389.
    #    This suggests the evaluator did NOT clamp to +/-1, or Insp 2 used a different window.
    #    Or maybe the "score 0.389" in the header is the SEARCH SCORE?
    #    If Search Score = 0.389, Raw Value = 1/0.389 = 2.57.
    #    All 1s has Raw Value 1.
    #    So Insp 2 is worse than All 1s?
    #    This is consistent if the window had high sidelobes.
    
    # Let's stick to the binary constraint. It's safer.
    # To get a signed window, we can use:
    # f[i] = 1 if i < N/2 else -1? (Step function)
    # Autocorrelation of a step function:
    # Lag 0: N
    # Lag k: N - 2k (for k < N/2)
    # Max sidelobe: N-1 (at lag 1).
    # Normalized: (N-1)/N ~ 1.
    # Bad.
    
    # What if we use a random sign sequence but initialize it with a structure?
    # Maybe a "chirp" or "Barker-like" sequence?
    # Barker sequences are optimal for small N. For N=400, we can't use a standard Barker.
    # But we can use a pseudo-random sequence with good autocorrelation properties, like a Zadoff-Chu sequence or a Gold code.
    # Gold codes are generated from two m-sequences.
    # Let's try initializing with a Gold code or a Zadoff-Chu sequence.
    # Zadoff-Chu: z[n] = exp(-j * pi * n * (n+1) / N) ? No, that's complex.
    # Real-valued Zadoff-Chu?
    # Or simply: f[n] = (-1)^{n(n+1)/2} ? (Thue-Morse sequence?)
    # Thue-Morse sequence has good autocorrelation properties.
    # Let's try Thue-Morse as an initial guess.
    
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n.
    # f[n] = 1 if t(n) == 0 else -1.
    
    # Let's add this as another restart initialization.
    
    # Revised Plan:
    # 1. Keep the local search.
    # 2. Add Thue-Morse sequence as a restart initialization.
    # 3. Add a "structured random" initialization: random signs but with a low-pass filter applied?
    #    No, that's continuous.
    # 4. Try optimizing a parametric binary sequence?
    #    e.g., f[n] = sign( cos(2 * pi * k * n / N) ) ?
    #    This is a square wave.
    #    Autocorrelation of square wave has high sidelobes.
    
    # Let's stick to the current program's approach but improve the initializations and maybe add a bit of simulated annealing to escape local minima.
    # The current program uses greedy local search.
    # Greedy can get stuck.
    # Simulated Annealing can escape.
    
    # I will combine:
    # - Multiple restarts with diverse initializations (Random, Thue-Morse, Best-So-Far).
    # - Greedy Local Search for quick improvement.
    # - Short Simulated Annealing phase to polish and escape local minima.
    
    # This is a structural change in the search method (adding SA) and representation (adding Thue-Morse).
    
    # Code structure:
    # 1. Define Thue-Morse generator.
    # 2. Define Local Search (Greedy).
    # 3. Define Simulated Annealing.
    # 4. Main loop:
    #    - For each restart:
    #      - Pick init (Random, Thue-Morse, or Best).
    #      - Run Greedy Local Search.
    #      - Run Simulated Annealing.
    #      - Update best.
    
    # Time budget: 140s.
    
    def _thue_morse(N):
        f = np.ones(N)
        for i in range(1, N):
            # t(i) = t(i//2) XOR (i%2)
            # But we need the parity.
            # Simpler: bin(i).count('1') % 2
            # This is slow for large N, but N=400 is small.
            pass
        # Vectorized version?
        # t(n) = sum of bits in n mod 2.
        # We can compute this using bit operations.
        # Or just use a recurrence.
        f = np.zeros(N)
        f[0] = 1.0
        for i in range(1, N):
            # t(i) = t(i >> 1) ^ (i & 1)
            # We need the actual value, not just parity for the recurrence?
            # No, t(i) is just the parity.
            # t(i) = t(i // 2) if i is even? No.
            # t(2k) = t(k)
            # t(2k+1) = 1 - t(k)
            # Let's use this.
            if i % 2 == 0:
                f[i] = f[i // 2]
            else:
                f[i] = 1.0 - f[i // 2]
        # Map 0/1 to -1/1
        return 2.0 * f - 1.0

    # Wait, the recurrence t(2k) = t(k) and t(2k+1) = 1-t(k) gives 0/1.
    # My f array stores 0/1?
    # f[0] = 1.0 (t(0)=0, so 1-0=1? No, t(0)=0. f[0] should be -1 if we map 0->-1, 1->1).
    # Let's just compute the binary string first.
    
    def _get_thue_morse(N):
        t = np.zeros(N, dtype=int)
        t[0] = 0
        for i in range(1, N):
            if i % 2 == 0:
                t[i] = t[i // 2]
            else:
                t[i] = 1 - t[i // 2]
        # Map to -1, 1
        return 1.0 - 2.0 * t

    # Let's test this mentally.
    # N=4
    # t[0]=0
    # t[1]=1-t[0]=1
    # t[2]=t[1]=1
    # t[3]=1-t[1]=0
    # Sequence: 0, 1, 1, 0
    # Map: 1, -1, -1, 1
    # Autocorrelation:
    # Lag 0: 4
    # Lag 1: 1*(-1) + (-1)*(-1) + (-1)*1 = -1 + 1 - 1 = -1
    # Lag 2: 1*(-1) + (-1)*1 = -1 - 1 = -2
    # Lag 3: 1*1 = 1
    # Max abs: 2. Normalized: 2/4 = 0.5.
    # Random for N=4 is likely ~0.5 as well.
    # For large N, Thue-Morse has low autocorrelation.
    
    # Let's implement the combined strategy.

    # EVOLVE-BLOCK-START
    import numpy as np
    import time
    import math

    N = 400

    def _max_sidelobe(f):
        """Compute max |autocorrelation| for lags 1..N-1, normalized by N."""
        n = len(f)
        nfft = 1
        while nfft < 2 * n:
            nfft *= 2
        
        F = np.fft.rfft(f, nfft)
        acf = np.fft.irfft(F * np.conj(F), nfft)[:n]
        if n > 1:
            return np.max(np.abs(acf[1:])) / n
        return 0.0

    def _thue_morse(N):
        t = np.zeros(N, dtype=int)
        t[0] = 0
        for i in range(1, N):
            if i % 2 == 0:
                t[i] = t[i // 2]
            else:
                t[i] = 1 - t[i // 2]
        return 1.0 - 2.0 * t

    def construct():
        start_time = time.time()
        deadline = start_time + 140.0
        
        rng = np.random.default_rng(42)
        
        # Initialize with random signs
        best_f = rng.choice([-1.0, 1.0], size=N)
        best_score = _max_sidelobe(best_f)
        
        # Also try Thue-Morse
        tm_f = _thue_morse(N)
        tm_score = _max_sidelobe(tm_f)
        if tm_score < best_score:
            best_score = tm_score
            best_f = tm_f.copy()
            
        max_restarts = 100
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Choose initialization
            if restart == 0:
                current_f = best_f.copy()
                current_score = best_score
            elif restart % 3 == 1:
                # Random
                current_f = rng.choice([-1.0, 1.0], size=N)
                current_score = _max_sidelobe(current_f)
            elif restart % 3 == 2:
                # Thue-Morse with random perturbation
                current_f = _thue_morse(N).copy()
                # Flip a few random bits
                for _ in range(10):
                    idx = rng.integers(0, N)
                    current_f[idx] *= -1.0
                current_score = _max_sidelobe(current_f)
            else:
                # Best so far
                current_f = best_f.copy()
                current_score = best_score
                
            if current_score < best_score:
                best_score = current_score
                best_f = current_f.copy()
                
            # Phase 1: Greedy Local Search
            improved = True
            max_passes = 5
            for _ in range(max_passes):
                if time.time() > deadline:
                    break
                if not improved:
                    break
                improved = False
                order = rng.permutation(N)
                for i in order:
                    if time.time() > deadline:
                        break
                    
                    current_f[i] *= -1.0
                    new_score = _max_sidelobe(current_f)
                    
                    if new_score < current_score - 1e-12:
                        current_score = new_score
                        improved = True
                        if current_score < best_score - 1e-12:
                            best_score = current_score
                            best_f = current_f.copy()
                    else:
                        current_f[i] *= -1.0
            
            # Phase 2: Simulated Annealing Polish
            if time.time() < deadline - 5.0:
                T = 0.01
                cooling = 0.99
                min_T = 1e-4
                
                while T > min_T and time.time() < deadline - 2.0:
                    idx = rng.integers(0, N)
                    current_f[idx] *= -1.0
                    new_score = _max_sidelobe(current_f)
                    
                    delta = new_score - current_score
                    if delta < 0 or rng.random() < math.exp(-delta / T):
                        current_score = new_score
                        if current_score < best_score - 1e-12:
                            best_score = current_score
                            best_f = current_f.copy()
                    else:
                        current_f[idx] *= -1.0
                        
                    T *= cooling
            
        return best_f
    # EVOLVE-BLOCK-END
