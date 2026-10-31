import numpy as np
import time
import itertools
from collections import defaultdict

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Build the Addition Collision Graph
    # Nodes: 0..N-1
    # Edge between i, j if there exist a, b (a!=b or (a,b) not proportional to (i,j) in a way that allows distinct sums?)
    # Actually, the directive says: "edge connects two indices if their sum collides with the sum of any other pair (including self-sums)"
    # This is slightly ambiguous. Let's interpret it as:
    # We want a set S such that the sums of elements in S are as unique as possible.
    # If i+j = k+l with {i,j} != {k,l}, then the autoconvolution at index i+j (or offset) gets a contribution from both pairs.
    # To minimize the peak, we want to avoid such collisions.
    # So, an edge should exist between i and j if the "sum structure" of {i,j} is not unique?
    # No, the graph nodes are indices. An edge (i,j) means "i and j are incompatible in the same set"?
    # If we include i and j, do they create a collision?
    # A collision happens if i+j = k+l. This involves 4 indices.
    # The MIS approach usually applies to a graph where edges represent conflicts.
    # If we define the graph G on vertices {0..N-1} such that (i,j) in E if there exists a pair (k,l) with i+j = k+l and {i,j} != {k,l}?
    # This definition is problematic because "incompatibility" is a hyperedge (4-way) relation, not a pairwise one.
    
    # Let's refine the interpretation based on "addition graph":
    # Perhaps the graph is defined on the *sums*? No, "maximum independent set search on the 'addition graph' of the index set".
    # Let's look at the property of Sidon sets (B2 sets).
    # A set S is Sidon if all sums a+b (a<=b) are distinct.
    # This is equivalent to saying that the only solution to a+b=c+d with a,b,c,d in S is {a,b}={c,d}.
    # This is a 4-uniform hypergraph property.
    # However, we can approximate this with a pairwise graph if we consider "differences" or specific sum collisions.
    
    # Alternative interpretation:
    # Maybe the "addition graph" has edges (i, j) if i+j is a "bad" sum?
    # Or maybe we construct a graph where nodes are potential *pairs*? No, MIS on index set.
    
    # Let's try a different structural constraint that is pairwise:
    # If we restrict the set S such that no two distinct pairs in S have the same sum, that's Sidon.
    # Can we encode Sidon-ness as an MIS problem on a graph of indices?
    # Not directly, because the conflict is 4-way.
    
    # However, we can use a greedy heuristic that *simulates* the MIS logic:
    # 1. Start with all indices.
    # 2. Iteratively remove the index that is involved in the most "sum collisions" with the remaining set.
    # This is a "Maximum Independent Set" heuristic on the 4-uniform hypergraph of sum collisions.
    # We can approximate the "degree" of a vertex i in this hypergraph by counting how many pairs (j,k) in the current set S satisfy i+j = k+l for some l?
    # Actually, for a fixed set S, the "badness" of adding i is determined by how many sums i+s (s in S) collide with existing sums s1+s2 (s1,s2 in S).
    
    # Let's implement a Greedy "Sum-Collision-Avoiding" Set Construction.
    # This is structurally different from simple Sidon (which checks pairwise differences) or continuous optimization.
    
    best_h = np.ones(N) / N
    best_score = compute_ratio(best_h)
    
    rng = np.random.default_rng(42)
    
    # Precompute all possible sums? No, N=50, sums go up to 98.
    # For a set S, we can maintain a count of sums.
    # sum_count[s] = number of pairs (a,b) in S with a<=b and a+b=s.
    # If we add x, new sums are x+s for s in S.
    # We want to minimize the increase in max(sum_count).
    
    def greedy_addition_graph_search(randomize_order=False):
        """
        Greedy construction of a subset S of {0..N-1} that minimizes additive ambiguity.
        """
        remaining = set(range(N))
        S = set()
        sum_counts = defaultdict(int)
        
        # If randomizing, shuffle the order of consideration
        candidates = list(remaining)
        if randomize_order:
            rng.shuffle(candidates)
        
        # We want to build a set of size ~10-20.
        # We will greedily add elements that cause the least "peak" in sum_counts.
        
        current_peak = 0
        
        # We don't know the target size, so we add until no more elements can be added without increasing the peak significantly?
        # Or we try to build a set of specific sizes and evaluate.
        # Let's try to build a maximal set where sum_counts[s] <= 2 for all s? (B2-like with some collisions allowed)
        # Actually, let's just greedily add the element that minimizes the *new* max sum count.
        
        # To allow for different "shapes", we can add a penalty for large indices or something?
        # No, let's just do the greedy min-peak addition.
        
        added_any = True
        while added_any:
            added_any = False
            best_candidate = None
            best_new_peak = current_peak + 1 # Default: any addition increases peak by at least 1? Not necessarily.
            
            # For each candidate, calculate the resulting peak
            # Optimization: only consider candidates that don't create a sum > 2*max(S)?
            # Sums range from 0 to 2(N-1).
            
            for x in candidates:
                if x in S:
                    continue
                
                # Calculate new sum counts
                # New sums: x + x, x + s for s in S
                # We need to check if these sums collide with existing high counts.
                
                # Fast check: 
                # If x+x or x+s already has count >= current_peak, it might increase the peak.
                # Let's just compute the max count after adding x.
                
                # To be efficient, we can pre-calculate the counts for x?
                # Since N=50 and |S| grows, this is O(|S|) per candidate.
                # Total O(N^2 * |S|) which is fine for N=50.
                
                max_count_after = current_peak
                # Check self-sum
                s_self = 2 * x
                c_self = sum_counts[s_self] + 1
                if c_self > max_count_after:
                    max_count_after = c_self
                
                # Check cross-sums
                for s in S:
                    s_cross = x + s
                    c_cross = sum_counts[s_cross] + 1
                    if c_cross > max_count_after:
                        max_count_after = c_cross
                
                if max_count_after < best_new_peak:
                    best_new_peak = max_count_after
                    best_candidate = x
            
            if best_candidate is not None:
                x = best_candidate
                S.add(x)
                # Update sum_counts
                sum_counts[2*x] += 1
                for s in S:
                    if s != x: # Avoid double counting x+x? 
                        # Wait, the loop above included s=x? No, s is from S which doesn't contain x yet.
                        # So x+x is handled separately.
                        # But x+s for s in S are all distinct from x+x if s!=x.
                        pass
                
                # Actually, let's update properly
                # We added x.
                # Pairs involving x: (x,x) and (x,s) for s in S (old S).
                sum_counts[2*x] += 1
                for s in S:
                    if s != x:
                        sum_counts[x+s] += 1
                
                current_peak = max(current_peak, best_new_peak)
                added_any = True
                # Remove from candidates list for efficiency? Or just check `if x in S` in loop.
                # The loop `for x in candidates` checks `if x in S`.
        
        # Convert to array
        h = np.zeros(N)
        for i in S:
            h[i] = 1.0
        h /= np.sum(h)
        return h

    # Run the greedy search with different randomizations
    # The greedy "min-peak" strategy is deterministic if we don't randomize tie-breaking.
    # Let's add random tie-breaking to explore different local optima.
    
    def greedy_with_random_ties(seed_offset=0):
        rng_local = np.random.default_rng(seed_offset)
        remaining = list(range(N))
        S = set()
        sum_counts = defaultdict(int)
        current_peak = 0
        
        while True:
            best_candidates = []
            best_new_peak = float('inf')
            
            for x in remaining:
                if x in S:
                    continue
                
                max_count_after = current_peak
                s_self = 2 * x
                c_self = sum_counts[s_self] + 1
                if c_self > max_count_after:
                    max_count_after = c_self
                
                for s in S:
                    s_cross = x + s
                    c_cross = sum_counts[s_cross] + 1
                    if c_cross > max_count_after:
                        max_count_after = c_cross
                
                if max_count_after < best_new_peak:
                    best_new_peak = max_count_after
                    best_candidates = [x]
                elif max_count_after == best_new_peak:
                    best_candidates.append(x)
            
            if not best_candidates:
                break
            
            # Pick one randomly from the best candidates
            x = rng_local.choice(best_candidates)
            S.add(x)
            
            sum_counts[2*x] += 1
            for s in S:
                if s != x:
                    sum_counts[x+s] += 1
            
            current_peak = best_new_peak
            
            # Optional: stop if peak gets too high? 
            # The ratio involves L2^2 / (L1 * Peak).
            # For equal weights h_i=1/k:
            # L2^2 = k*(1/k)^2 = 1/k
            # L1 = 1
            # Peak = max(sum_counts)
            # Ratio = (1/k) / (1 * Peak) = 1/(k * Peak)
            # To maximize ratio, we want to minimize k * Peak.
            # As k increases, 1/k decreases. But Peak might stay low.
            # If Peak=1, Ratio = 1/k. Decreasing.
            # If Peak=2, Ratio = 1/(2k).
            # So we want small k and small Peak.
            # However, if we allow larger k with Peak=1, ratio drops.
            # Wait, if h is not equal weights, we can adjust weights.
            # But for the support search, equal weights is a good baseline.
            
            # Let's just build the maximal set for this greedy path.
            
        h = np.zeros(N)
        for i in S:
            h[i] = 1.0
        h /= np.sum(h)
        return h

    # Try a few random restarts of the greedy addition graph search
    for i in range(20):
        if time.time() > deadline:
            break
        h_candidate = greedy_with_random_ties(seed_offset=i)
        score = compute_ratio(h_candidate)
        if score > best_score:
            best_score = score
            best_h = h_candidate.copy()

    # 2. Local optimization of weights on the best support found
    if np.count_nonzero(best_h) > 0:
        support_idx = np.nonzero(best_h)[0]
        k = len(support_idx)
        
        if k > 0:
            def sparse_neg_ratio(w):
                h = np.zeros(N)
                h[support_idx] = np.maximum(w, 0)
                s = np.sum(h)
                if s < 1e-12: return 1e6
                h /= s
                return -compute_ratio(h)
            
            w0 = np.ones(k) / k
            
            try:
                from scipy.optimize import minimize
                # Use Nelder-Mead for robustness
                res = minimize(sparse_neg_ratio, w0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < -best_score:
                    w_opt = np.maximum(res.x, 0)
                    s = np.sum(w_opt)
                    if s > 1e-12:
                        w_opt /= s
                        h_opt = np.zeros(N)
                        h_opt[support_idx] = w_opt
                        score = compute_ratio(h_opt)
                        if score > best_score:
                            best_score = score
                            best_h = h_opt
            except Exception:
                pass

    # 3. If time permits, try a few "Sidon-like" sets generated by different heuristics
    # E.g., greedy by index order
    if time.time() < deadline - 20:
        # Greedy by index order (deterministic)
        h_det = greedy_with_random_ties(seed_offset=-1) # This will just pick the first best candidate if no ties? 
        # Actually, rng_local.choice with a single candidate returns that candidate.
        # If multiple, it picks randomly. 
        # For deterministic, we can just pick the first one in the list.
        # Let's just evaluate the deterministic one (seed 0 usually gives a specific path)
        pass

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
