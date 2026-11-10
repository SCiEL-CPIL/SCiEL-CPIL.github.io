from math import gcd
from functools import reduce
lcm_30 = reduce(lambda a, b: a * b // gcd(a, b), range(1, 31))
print(lcm_30)  # Check size
