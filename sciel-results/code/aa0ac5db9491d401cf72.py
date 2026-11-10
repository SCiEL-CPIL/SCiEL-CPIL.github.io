<<<<<<< SEARCH
import time
import random
from itertools import combinations

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    sumset = set()
    prodset = set()
    for i in range(len(A)):
        for j in range(len(A)):
            sumset.add(A[i] + A[j])
            prodset.add(A[i] * A[j])
    return max(len(sumset), len(prodset))
=======
import time
import random
import numpy as np

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    arr = np.array(A, dtype=np.int64)
    sums = np.add.outer(arr, arr)
    prods = np.multiply.outer(arr, arr)
    sumset_size = len(np.unique(sums))
    prodset_size = len(np.unique(prods))
    return max(sumset_size, prodset_size)
>>>>>>> REPLACE

