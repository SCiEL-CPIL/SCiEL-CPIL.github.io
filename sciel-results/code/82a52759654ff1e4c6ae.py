import random
import time

def construct():
    n = 18
    # Initialize with Turan graph T(18,3) or random
    # Let's start with random and enforce K4-free
    # Or start with T(18,3) and perturb?
    # T(18,3) is a local minimum for score?
    # Maybe start with a random graph with high edge prob, then remove K4s.
    
    # Time limit 120s.
    # We need an efficient score calculation.
