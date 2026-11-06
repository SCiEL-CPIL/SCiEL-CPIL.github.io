import time

def construct():
    # The set {7, 8, ..., 24} is valid and has size 18.
    # The seed program returns {13, ..., 24} with size 12.
    # Score improvement: 18/12 = 1.5.
    return list(range(7, 25))
