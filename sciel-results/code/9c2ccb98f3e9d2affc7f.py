import time
import random
import math

def construct():
    deadline = time.time() + 115
    
    def time_left():
        return deadline - time.time()
        
    # ... SA logic ...
    
    # 1. Find n=26
    # Run SA for n=26 with time limit 5s
    # If found, save.
    
    # 2. Try n=27
    # Initialize from n=26 solution + random vertex
    # Run SA for remaining time
    
    # Return best
