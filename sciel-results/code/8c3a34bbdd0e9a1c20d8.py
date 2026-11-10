import random


def construct():
       best_set = []
       
       def try_order(order):
           S = set()
           sums = {0}
           for x in order:
               if (200 - x) not in sums:
                   S.add(x)
                   new_sums = set()
                   for s in sums:
                       ns = s + x
                       if ns <= 200:
                           new_sums.add(ns)
                   sums.update(new_sums)
           return S
       
       # Descending
       S1 = try_order(range(100, 0, -1))
       
       # Ascending
       S2 = try_order(range(1, 101))
       
       # Random
       random.seed(42)
       best_rand = set()
       for _ in range(10):
           order = list(range(1, 101))
           random.shuffle(order)
           Sr = try_order(order)
           if len(Sr) > len(best_rand):
               best_rand = Sr
               
       candidates = [S1, S2, best_rand]
       best = max(candidates, key=len)
       
       return list(best)
