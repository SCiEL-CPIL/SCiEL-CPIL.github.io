<<<<<<< SEARCH
import time
import random
import math
import numpy as np

def _score(A):
    """Compute max(|A+A|, |A*A|) using numpy for speed."""
    arr = np.array(A, dtype=np.int64)
    sums = np.add.outer(arr, arr)
    prods = np.multiply.outer(arr, arr)
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)
=======
import time
import random
import math
import numpy as np

def _score(A):
    """Compute max(|A+A|, |A*A|) using numpy for speed."""
    arr = np.array(A, dtype=np.int64)
    sums = np.add.outer(arr, arr)
    prods = np.multiply.outer(arr, arr)
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def _score_detailed(A):
    """Return (max_score, n_sums, n_prods) for analysis."""
    arr = np.array(A, dtype=np.int64)
    sums = np.add.outer(arr, arr)
    prods = np.multiply.outer(arr, arr)
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods), n_sums, n_prods
>>>>>>> REPLACE

