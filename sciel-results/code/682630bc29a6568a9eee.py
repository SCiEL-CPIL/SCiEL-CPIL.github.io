"""Cap set greedy insertion with algebraic-degree and CRC32-based priority."""
import time
import zlib
import numpy as np

# EVOLVE-BLOCK-START
def priority(vector, n):
    # vector is a tuple of {0,1,2} of length n
    # We want to prioritize vectors that are "algebraically rich" (higher degree)
    # and break ties with a fast hash (CRC32) to encourage diversity/structure.
    
    # 1. Algebraic degree heuristic:
    # Treat vector as coefficients of a polynomial in F_3[x].
    # The degree is the index of the highest non-zero coefficient.
    # Higher degree => more "non-linear" structure, potentially harder to form lines.
    # However, simply maximizing degree might lead to sparse high-degree vectors.
    # Let's combine degree with a count of non-zero terms or a weighted sum.
    
    # A stronger heuristic for cap sets:
    # Cap sets in F_3^n are related to avoiding solutions to x+y=z (affine lines).
    # In the polynomial interpretation, affine lines correspond to linear dependencies.
    # Vectors with many non-zero entries of varying "frequencies" might be better.
    
    # Let's try a score based on the sum of (value * weight) where weight increases with index?
    # Or simply the degree?
    
    # The prompt suggests: "selecting those with higher-order non-zero terms"
    # So, prioritize higher degree.
    
    # Calculate degree:
    degree = 0
    for i in range(n - 1, -1, -1):
        if vector[i] != 0:
            degree = i
            break
    
    # To break ties and add some pseudo-randomness/structure, use CRC32.
    # Convert vector to bytes for CRC32.
    # Since n can be small, we can just pack the values.
    # Values are 0, 1, 2. We can use 2 bits each, or just 1 byte each.
    # 1 byte each is simpler and fast enough for small n.
    # For large n, we might need to be careful, but cap sets are usually small n.
    
    # Let's use a simple hash for tie-breaking.
    # The observed change said CRC32 is faster and yielded 1.039 -> 1.09.
    
    # We need a deterministic priority.
    # priority = degree * large_number + hash_value?
    # Or just use degree as the primary sort key, and hash as secondary?
    # The function returns a single float.
    
    # If we return degree, vectors with same degree are tied.
    # The greedy algorithm might then pick them in input order or arbitrary order.
    # To make it deterministic and structured, we mix in the hash.
    
    # Let's compute CRC32 of the vector bytes.
    # vector is a tuple of ints.
    # We can create a bytes object from the vector.
    # Since values are 0-2, they fit in 1 byte.
    
    # Optimization: Pre-compute? No, priority is called per vector.
    # Is priority called for every vector in {0,1,2}^n?
    # If n is large, 3^n is huge. But the task says "vectors ... are added greedily".
    # Usually, the evaluator generates the list of all vectors, sorts by priority, and tries to add them.
    # If n is e.g. 8, 3^8 = 6561. 3^10 = 59049. 3^12 = 531441.
    # 3^15 = 14M.
    # If n is large, calling priority 14M times might be slow if it's complex.
    # CRC32 is fast. Degree calculation is O(n).
    
    # Let's refine the degree heuristic.
    # Maybe "weight" the degree more?
    # Or use the sum of indices of non-zero terms?
    
    # Let's stick to the prompt's suggestion: "higher-order non-zero terms".
    # So degree is the main factor.
    
    # To ensure we get a good spread, let's use:
    # score = degree * 1e6 + (crc32 % 1e6)
    
    # Wait, if we prioritize high degree, we pick vectors like (0,0,...,0,1) first.
    # These are very sparse.
    # Cap sets often benefit from more "dense" vectors?
    # Actually, the seed (uniform) gave 1.0.
    # The CRC32 only (random-like) gave 1.09.
    # This suggests that *random* ordering is better than lexicographic.
    # Maybe algebraic degree is too biased?
    
    # Let's look at the "Inspiration" again:
    # "Prioritize vectors based on their algebraic degree ... which enforces a structured non-linearity ... resistant to linear dependencies"
    
    # If random (CRC32) is 1.09, and uniform is 1.0, then structure might help if it's the *right* structure.
    # But maybe pure degree is bad?
    
    # Let's try a combination:
    # 1. Randomness (CRC32) is good.
    # 2. Maybe add a small bias for degree?
    
    # Or, perhaps the "algebraic degree" should be interpreted differently.
    # What if we use the *number* of non-zero terms?
    # Or the *sum* of the values?
    
    # Let's try: priority = crc32 + 0.001 * degree
    # This keeps the random ordering dominant but breaks ties slightly by degree.
    # Or, since random was 1.09, maybe we should stick close to random but add a specific structure.
    
    # Actually, the prompt says "Strategy to try ... Prioritize vectors based on their algebraic degree".
    # So I should follow that strategy, but maybe refine it.
    
    # If I just use degree, I get a very specific order.
    # Let's try:
    # score = degree * 1000 + (crc32 & 0xFF)
    
    # Let's consider the cost.
    # If n is large, 3^n is large.
    # We need to be fast.
    
    # Implementation:
    # 1. Find degree.
    # 2. Compute CRC32.
    # 3. Combine.
    
    # To make it fast, avoid creating a bytes object if possible?
    # zlib.crc32 accepts bytes.
    # vector is a tuple.
    # bytes(vector) works if values are 0-255.
    
    # Let's try this.
    
    # One concern: if n is very large, degree calculation is O(n).
    # We can optimize by checking from the end.
    
    # Let's also consider that the evaluator might call this many times.
    # We should cache if possible? No, vector is the key.
    
    # Let's write the function.
    
    # Fast degree calculation
    deg = 0
    # Iterate backwards
    for i in range(n - 1, -1, -1):
        if vector[i] != 0:
            deg = i
            break
    
    # Fast CRC32
    # Convert to bytes
    # This allocation might be the bottleneck if called millions of times.
    # Alternative: manual CRC32? No, zlib is C-optimized.
    # Can we avoid bytes conversion?
    # zlib.crc32 requires bytes.
    
    # If n is small, bytes(vector) is fine.
    
    h = zlib.crc32(bytes(vector))
    
    # Combine
    # We want degree to be the primary driver?
    # Or random?
    # The prompt says "Strategy to try ... Prioritize ... based on algebraic degree".
    # So degree should be significant.
    
    # Let's return: deg * 1e9 + h
    # This ensures all degree 0 vectors come before degree 1, etc.
    # Within same degree, sorted by CRC32.
    
    # Is this a good strategy?
    # Degree 0: (0,...,0) -> 1 vector
    # Degree 1: (0,...,0,1), (0,...,0,2) -> 2 vectors
    # Degree 2: (0,...,0,1,1), ... -> 3*2 vectors?
    # Actually, number of vectors with degree exactly k is 2 * 3^(n-k-1).
    # So low degree vectors are few, high degree vectors are many.
    # If we prioritize high degree, we pick from the largest group first.
    # If we prioritize low degree, we pick the sparse ones first.
    
    # "Higher-order non-zero terms" implies high degree.
    
    # Let's go with high degree first.
    
    return deg * 1e9 + h

# EVOLVE-BLOCK-END
