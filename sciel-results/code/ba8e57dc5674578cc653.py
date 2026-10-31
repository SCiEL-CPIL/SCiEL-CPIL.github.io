import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Worst-Fit strategy:
    # We want the bin with the LARGEST remaining capacity among valid bins.
    # This spreads items out, keeping bins "full" rather than "partially used",
    # which can be beneficial for certain item distributions.
    # To achieve this with a "highest score wins" mechanism, we assign
    # scores proportional to the remaining capacity.
    # score = remaining
    # Largest remaining -> Highest score.
    # Smallest remaining -> Lowest score.
    
    # Invalid bins get -inf so they are never selected.
    remaining = bins - item
    valid = remaining >= 0
    scores = np.where(valid, remaining, -np.inf)
    
    return scores
