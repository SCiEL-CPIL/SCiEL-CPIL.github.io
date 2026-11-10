import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    # Generate all 2^11 binary vectors
    vectors = []
    for i in range(2**d):
        # Convert integer to binary vector of length d
        bin_str = format(i, f'0{d}b')
        # Map 0 to -1, 1 to 1
        vec = np.array([1 if c == '1' else -1 for c in bin_str], dtype=np.int64)
        vectors.append(vec)
    
    # Greedy selection
    selected = []
    for v in vectors:
        valid = True
        for u in selected:
            # Hamming distance is number of differing bits
            # In +/- 1 representation, dist^2 = 2 * Hamming Distance
            # We need dist^2 >= 11 => 2 * Hamming >= 11 => Hamming >= 5.5 => 6
            diff = np.sum(v != u)
            if diff < 6:
                valid = False
                break
        if valid:
            selected.append(v)
            
    return np.array(selected)
