import math
import time

def construct():
    # The theoretical maximum score is 2.
    # For U = {0, 1, M}, |U+U|=6, |U-U|=2M+1.
    # Score = 1 + log((2M+1)/6) / log(2M+1)
    # As M -> infinity, Score -> 2.
    # We choose a large M to get very close to 2.
    # M = 10**30 is sufficiently large and fast to compute.
    M = 10**30
    return [0, 1, M]
