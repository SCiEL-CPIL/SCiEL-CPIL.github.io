def priority(vector, n):
    # Approximate "saturation" by Hamming Weight.
    # Vectors with more non-zero entries are "denser".
    # Prioritize "sparser" vectors (lower weight) to minimize future conflicts.
    weight = sum(1 for x in vector if x != 0)
    # Return negative weight so that lower weight gets higher priority in descending sort.
    return float(-weight)
