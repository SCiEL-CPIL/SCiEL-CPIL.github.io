S(1) = [1]
S(n) from S(n-1):
  result = S(n-1)
  for each permutation p of {1,...,n-1}:
    find p in result (as contiguous subsequence)
    insert n before the first occurrence of p
    append to result: n, then the elements of p in the order they appear, but starting from the second element and wrapping: p[1], p[2], ..., p[n-2], p[0]
