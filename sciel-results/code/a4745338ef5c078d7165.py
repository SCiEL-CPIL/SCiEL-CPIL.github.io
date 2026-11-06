   def merge(A, B):
       # Merges two sorted lists of size A and B
       # Returns list of comparators
       if B == 0: return []
       if A == 0: return []
       
       # If A == B, use Batcher's merge
       # If A != B, split the larger one?
       
       # Simplest: Use Odd-Even Merge if A and B are powers of 2?
       # For A=16, B=1:
       # We can use a specific pattern.
   