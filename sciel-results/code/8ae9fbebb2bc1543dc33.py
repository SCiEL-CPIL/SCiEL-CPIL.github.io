import random
import time

def construct():
    # Use a structured Arithmetic Progression which minimizes the sumset size (2n-1).
    # While the productset is larger than the sumset, this is typically better than 
    # random sets or geometric progressions (which have huge sumsets) under the 
    # max(|A+A|, |A*A|) objective for n=20 and max_val=10000.
    
    A = list(range(1, 21))
    
    return A
