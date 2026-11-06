"""
Strategy:
1. Use a greedy forward construction approach to build the sorting network.
2. At each step, evaluate candidate comparators based on how much they reduce the average inversion count across a set of sample inputs (including all 0/1 inputs if possible, or a large random subset).
3. Since N=16, the total number of 0/1 inputs is 65,536. We can process all of them simultaneously using vectorized NumPy operations to get an exact "inversion reduction" score for each candidate comparator.
4. This avoids the dependency on Batcher's network structure and directly searches for a minimal network.
5. To ensure correctness and handle the 60s limit, we maintain a deadline. If we run out of time, we return the best valid network found so far.
6. We start with the empty network (or a small seed) and add comparators one by one.
7. We use the "Zero-One Principle" implicitly by checking all 0/1 inputs at the end, but the greedy choice is driven by the inversion count on these inputs.
"""

import time
import numpy as np

N = 16
MAX_TIME = 55.0

def _generate_all_01_inputs():
    """
    Generate all 2^N 0/1 input vectors as a NumPy array of shape (2^N, N).
    """
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.int64)
    # We want bit 0 to be the rightmost (index N-1) or leftmost?
    # Standard: index i in the array corresponds to wire i.
    # Let's map input value v to vector v.
    # v = sum(b_i * 2^i)
    # We want vec[:, i] = b_i.
    # So vec[:, i] = (v >> i) & 1.
    bits = (indices[:, None] >> np.arange(N)) & 1
    return bits

def _count_inversions(vecs):
    """
    Count the total number of inversions in each vector.
    An inversion is a pair (i, j) with i < j such that vecs[k, i] > vecs[k, j].
    Returns an array of shape (num_vecs,) with the inversion counts.
    """
    # For binary vectors, inversion count is sum over i<j of (1 - vecs[i]) * vecs[j]
    # Or more efficiently: for each vector, count how many 1s appear after each 0?
    # Actually, for binary:
    # Inversion at (i, j) exists if vecs[i]=1 and vecs[j]=0.
    # Total inversions = sum_{j} (number of 1s in positions < j) * vecs[j] == 0? No.
    # Total inversions = sum_{i} (number of 0s in positions > i) * (1 - vecs[i])?
    
    # Let's use a simpler vectorized approach for small N.
    # N=16 is small.
    # inv_count[k] = sum_{i < j} (vecs[k, i] > vecs[k, j])
    # Since binary, vecs[k, i] > vecs[k, j] iff vecs[k, i] == 1 and vecs[k, j] == 0.
    
    # We can compute this by iterating over j and counting 1s before j?
    # Or using matrix multiplication?
    # Let A be the matrix of 1s.
    # Inversion count for row k is sum_{i<j} A[k,i] * (1-A[k,j]).
    
    # Efficient way:
    # For each column j, let prefix_sum_ones[:, j] be the number of 1s in columns 0..j-1.
    # Then inversions contributed by column j being 0 is prefix_sum_ones[:, j].
    # Wait, if vecs[j] is 0, it forms an inversion with every 1 before it.
    # So inv_count = sum_j (1 - vecs[:, j]) * prefix_sum_ones[:, j]
    
    ones = vecs
    prefix_ones = np.cumsum(ones, axis=1)
    # prefix_ones[:, j] contains sum of ones in cols 0..j.
    # We need sum of ones in cols 0..j-1.
    # So shift right by 1, padding 0 at start.
    shifted_prefix = np.zeros_like(prefix_ones)
    shifted_prefix[:, 1:] = prefix_ones[:, :-1]
    
    zeros = 1 - vecs
    inv_counts = np.sum(zeros * shifted_prefix, axis=1)
    return inv_counts

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Generate all 0/1 inputs
    all_inputs = _generate_all_01_inputs()
    
    # Current network state
    current_comps = []
    
    # Initial state: all inputs
    current_vecs = all_inputs.copy()
    
    # Calculate initial inversion counts
    current_inv = _count_inversions(current_vecs)
    
    # Target: 0 inversions for all inputs.
    # We will add comparators one by one.
    # Max comparators allowed is 4096, but we want minimal.
    # We stop when all inputs are sorted (0 inversions).
    
    # To speed up, we can restrict candidates?
    # For N=16, there are 16*15/2 = 120 possible comparators.
    # We can evaluate all 120 at each step?
    # 120 candidates * (cost of applying 1 comp + counting inversions).
    # Applying 1 comp to 65536x16 array is fast.
    # Counting inversions is O(N^2 * NumInputs) = 16^2 * 65536 ~ 16M ops.
    # 120 * 16M = 2B ops per step. This might be too slow in Python/NumPy if not optimized.
    
    # Optimization:
    # Instead of counting full inversions, we can just check if the network is sorted?
    # No, we need a gradient.
    
    # Alternative:
    # We can use a smaller sample set for the greedy step, and verify with the full set at the end.
    # Let's try with a subset first to see speed.
    # If N=16, 65536 inputs is manageable in NumPy.
    # Let's profile the inversion count.
    
    # Actually, there is a faster way to check if a comparator (i, j) reduces inversions.
    # It reduces inversions if and only if there exists an input where wire i > wire j.
    # But we want to maximize reduction.
    
    # Let's stick to the full inversion count but optimize the loop.
    # We can pre-compute the "potential" of each comparator?
    
    # Let's try a different metric: "Disorder".
    # For each input, the number of pairs (i, j) with i < j and vec[i] > vec[j].
    
    # Let's implement the greedy step carefully.
    
    while True:
        if time.time() > deadline - 2.0:
            break
        
        # Check if sorted
        if np.all(current_inv == 0):
            break
        
        best_comp = None
        best_score = -1
        best_new_inv = None
        
        # Candidate comparators: all (i, j) with i < j
        # We can filter out comparators that are "no-ops" on the current state?
        # A comparator (i, j) is a no-op if vec[i] <= vec[j] for ALL inputs.
        
        candidates = []
        for i in range(N):
            for j in range(i + 1, N):
                # Check if this comparator does anything
                # If current_vecs[:, i] <= current_vecs[:, j] for all rows, it's a no-op.
                # We can check this quickly.
                if np.all(current_vecs[:, i] <= current_vecs[:, j]):
                    continue
                candidates.append((i, j))
        
        if not candidates:
            # Stuck? Should not happen if not sorted, but just in case.
            # Force a random comparator or break.
            break
        
        # Evaluate each candidate
        # To save time, we can batch process or just loop.
        # 120 candidates max.
        
        for i, j in candidates:
            # Apply comparator
            new_vecs = current_vecs.copy()
            min_val = np.minimum(new_vecs[:, i], new_vecs[:, j])
            max_val = np.maximum(new_vecs[:, i], new_vecs[:, j])
            new_vecs[:, i] = min_val
            new_vecs[:, j] = max_val
            
            # Calculate new inversion count
            # Optimization: We only need to check if the total inversion count decreased.
            # Actually, we want the one with the LOWEST total inversion count.
            new_inv = _count_inversions(new_vecs)
            total_inv = np.sum(new_inv)
            
            if total_inv < best_score or best_score == -1:
                best_score = total_inv
                best_comp = (i, j)
                best_new_inv = new_inv
                best_new_vecs = new_vecs
        
        if best_comp is None:
            break
            
        # Commit the best comparator
        current_comps.append(best_comp)
        current_inv = best_new_inv
        current_vecs = best_new_vecs
        
        # Optional: Prune redundant comparators?
        # For now, just build.
        
    # Final Verification
    # The greedy process ensures 0 inversions on the 0/1 inputs used.
    # Since we used ALL 0/1 inputs, the network is correct by the Zero-One Principle.
    
    # However, we might have added redundant comparators.
    # We can try to remove them.
    
    # Attempt to reduce size
    for i in range(len(current_comps) - 1, -1, -1):
        if time.time() > deadline - 1.0:
            break
        
        candidate = current_comps[:i] + current_comps[i+1:]
        
        # Verify candidate
        test_vecs = all_inputs.copy()
        for a, b in candidate:
            min_val = np.minimum(test_vecs[:, a], test_vecs[:, b])
            max_val = np.maximum(test_vecs[:, a], test_vecs[:, b])
            test_vecs[:, a] = min_val
            test_vecs[:, b] = max_val
        
        # Check if sorted
        # Instead of counting inversions, just check if non-decreasing
        # diffs = np.diff(test_vecs, axis=1)
        # if np.all(diffs >= 0):
        
        # Faster check:
        # For binary, sorted means all 0s then all 1s.
        # We can check if the vector is of form 0...01...1
        # This is equivalent to checking if for all k, if test_vecs[k, j] == 1 then test_vecs[k, i] == 1 for all i < j?
        # No, that's not right.
        # Sorted means: for all i < j, test_vecs[i] <= test_vecs[j].
        
        # Let's use the inversion count again, it's robust.
        test_inv = _count_inversions(test_vecs)
        if np.all(test_inv == 0):
            current_comps = candidate
    
    return current_comps

if __name__ == "__main__":
    # Quick test
    net = construct()
    print(f"Network size: {len(net)}")
