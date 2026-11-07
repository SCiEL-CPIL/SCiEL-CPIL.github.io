<<<<<<< SEARCH
"""Direct construction: find a maximal set of distinct integers in 2..40
whose 2^k subset products are all distinct.

Key structural insight: if we use only primes, subset products are distinct
because of unique factorization. There are 12 primes in 2..40:
2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37

But we can also add some composites if they don't create collisions.
A composite c = p1^a1 * p2^a2 * ... can be added if no existing subset
product equals c times another existing subset product (i.e., c is not a
quotient of two existing subset products, and c is not already a subset
product).

Actually, the condition for adding x to a set S is:
  For all a, b in subset_products(S), a*x != b  (i.e., a*x not in prods)
  Equivalently: x is not in {b/a : a,b in prods, a|x... } -- complex.

Simpler direct approach: use all 12 primes, then greedily try to add
each composite (4,6,8,9,10,12,14,15,16,18,20,21,22,24,25,26,27,28,30,
32,33,34,35,36,38,39,40) in some order, checking the distinctness
condition each time.

To maximize the count, we should try multiple orders and use local search.
But the direct structural insight is: primes are always safe, so start
with all 12 primes, then add composites that fit.

We can get up to 12 + (composites that fit). Let's compute which composites
can be added after all primes.

After all primes, subset products are all products of subsets of the 12
primes, i.e., all square-free numbers whose prime factors are from our set.
A composite c can be added iff c is not a ratio of two such products, i.e.,
c is not a square-free number composed of our primes (since c = b/a where
a,b are square-free with primes from our set means c is also square-free
with primes from our set).

Wait, that's not quite right. a*x = b where a,b are in prods. If x is a
composite, say x = p^2, then a*p^2 = b. Since b is square-free, this is
impossible unless a=0 which isn't in prods. So p^2 can always be added?

Let me think more carefully. prods contains all products of subsets of the
12 primes. These are all square-free numbers (products of distinct primes
from our set). So prods = {product of subset of {2,3,5,...,37}}.

If we try to add x = 4 = 2^2: we need to check if a*4 = b for any a,b in
prods. a*4 = b means b is divisible by 4, but b is square-free, so b is not
divisible by 4. Thus a*4 is never in prods (unless a=0, which isn't there).
So 4 can be added!

Similarly, x = p^2 for any prime p in our set: p^2 * a = b implies b
divisible by p^2, but b is square-free, contradiction. So all prime squares
can be added.

What about x = 6 = 2*3? a*6 = b. a is square-free with primes from our set.
a*6 = b means b = 6*a. Since a may already contain 2 or 3, 6*a may not be
square-free. If a doesn't contain 2 or 3, then b = 6*a is square-free and
in prods (since 2,3,a's primes are all in our set). So 6 CANNOT be added.

So the rule: a composite x can be added iff x is not a square-free number
composed solely of primes from our set. Equivalently, x has a prime power
p^k with k>=2, or x has a prime factor not in our set (but all primes in
2..40 are in our set since we have all 12 primes).

So composites that are NOT square-free (i.e., have a repeated prime factor)
can be added:
4=2^2, 8=2^3, 9=3^2, 12=2^2*3, 16=2^4, 18=2*3^2, 20=2^2*5, 24=2^3*3,
25=5^2, 27=3^3, 28=2^2*7, 32=2^5, 36=2^2*3^2, 40=2^3*5

Let me list all composites in 2..40 and mark which are square-free:
Square-free composites (CANNOT be added): 6,10,14,15,21,22,26,33,34,35,38,39
Non-square-free composites (CAN be added): 4,8,9,12,16,18,20,24,25,27,28,32,36,40

That's 14 composites that can be added! 12 primes + 14 = 26, but max is 18.

So we can definitely get 18. Let's construct: all 12 primes + 6 of the
non-square-free composites = 18.

But wait, after adding some composites, can we still add others? Let's think:
After adding 4, prods now includes products with 4. Can we still add 8?
We need a*8 not in prods for all a in prods. prods now includes 4, so
4*8=32. Is 32 in prods? prods includes 32 if 32 is a subset product. 32
is not a subset product of primes (it's 2^5, not square-free). But 32 might
be in prods if we added 32 earlier. We haven't. So 32 is not in prods yet.

Actually, this is getting complex. Let me just use a direct greedy approach
starting with primes, then adding non-square-free composites, checking the
condition each time.

Since the task says max 18, and we can clearly get at least 12 (primes) and
likely 18, let me just build a set of 18 directly and verify.

Direct construction: 12 primes + 6 composites from {4,8,9,12,16,18,20,24,
25,27,28,32,36,40}. Let's pick 6 that work together.

Actually, let me just write code that does this efficiently.
"""
import time

_P0 = time.time()
_DEADLINE = _P0 + 80.0  # safety margin under 90s
=======
"""Find a maximal set of distinct integers in 2..40 whose 2^k subset
products are all distinct. Uses exponent-vector independence check and
branch-and-bound to find 18 elements."""
import time
from itertools import combinations

_P0 = time.time()
_DEADLINE = _P0 + 75.0  # safety margin under 90s
>>>>>>> REPLACE

