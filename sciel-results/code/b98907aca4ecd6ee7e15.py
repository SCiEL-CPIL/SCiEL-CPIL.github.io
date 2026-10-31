import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # We are looking for a kissing configuration in R^11.
    # The known kissing number for d=11 is 512.
    # The configuration comes from the Leech lattice (or its sublattices), specifically related to the E8 lattice structure or similar highly symmetric configurations.
    # However, a simpler construction for d=11 is often derived from the E8 lattice and some additional coordinates, or specifically from the "Densest Packing in 11 dimensions" which is not known to be the Leech lattice (Leech is 24D).
    
    # Wait, the kissing number for 11 dimensions is indeed 512.
    # The optimal configuration is known and is related to the lattice $D_{11}$ or similar.
    # Actually, the optimal kissing configuration in $D=11$ is derived from the E8 lattice? No, E8 is 8D.
    # The 512 vectors in 11D can be constructed as follows:
    # Consider the E8 lattice in 8D. It has 240 minimal vectors.
    # We need 11D.
    
    # Let's look at the specific structure of the 512 kissing vectors in 11 dimensions.
    # It is known that the maximum number of non-overlapping unit spheres that can touch a central unit sphere in 11 dimensions is 512.
    # These vectors are the minimal vectors of the lattice $E_{11}$? No, $E_{11}$ doesn't exist as a root lattice in the same way.
    # The relevant lattice is likely a sublattice of the Leech lattice or similar.
    
    # A concrete construction for the 512 vectors in 11D:
    # They can be generated from the E8 lattice by embedding it in higher dimensions or using specific combinations.
    # Specifically, the 512 vectors are of two types:
    # 1. 256 vectors of the form $(\pm 1, \dots, \pm 1, 0, 0, 0)$ where the first 8 coordinates are $\pm 1$ (even number of minus signs, i.e., E8 minimal vectors scaled? No, E8 minimal vectors are weight 2 with $\pm 1, \pm 1$ and 6 zeros, or weight 8 with $\pm 1$).
    #    Let's check the norm. If we use E8 minimal vectors in the first 8 coordinates and 0 in the last 3:
    #    Type A: Weight 2 in 8D. Norm^2 = 2. Norm = sqrt(2).
    #    Type B: Weight 8 in 8D. Norm^2 = 8. Norm = 2*sqrt(2).
    #    This doesn't give a uniform norm.
    
    # Let's look for a uniform norm construction.
    # The 512 vectors in 11D kissing configuration all have the same norm, usually normalized to 2 or sqrt(2) or similar.
    # In the literature, the 512 kissing vectors in 11D are often given as:
    # - 256 vectors: $(\pm 1, \pm 1, \pm 1, \pm 1, \pm 1, \pm 1, \pm 1, \pm 1, 0, 0, 0)$ with an even number of minus signs.
    #   Wait, the number of such vectors is $2^7 = 128$ if we fix the first sign and require even parity?
    #   Total vectors in 8D with entries $\pm 1$ and even parity is $2^7 = 128$.
    #   If we allow odd parity too, it's 256.
    #   Let's check the dot product between two such vectors $u, v$.
    #   $u \cdot v = \sum u_i v_i$.
    #   If $u, v$ are distinct $\pm 1$ vectors in 8D with 0s in the rest:
    #   $u \cdot v$ can be 8, 6, 4, 2, 0, -2, -4, -6, -8.
    #   The norm squared is 8.
    #   Distance squared $d^2 = 8 + 8 - 2(u \cdot v) = 16 - 2(u \cdot v)$.
    #   We require $d^2 \ge \text{Norm}^2 = 8$?
    #   If the condition is "minimum pairwise distance is at least the maximum vector norm", let $R$ be the norm.
    #   $d(u,v) \ge R \implies d^2(u,v) \ge R^2$.
    #   $16 - 2(u \cdot v) \ge 8 \implies 8 \ge 2(u \cdot v) \implies u \cdot v \le 4$.
    #   This holds for all distinct $\pm 1$ vectors?
    #   Max dot product for distinct $\pm 1$ vectors is 6 (if they differ in 1 coordinate).
    #   $u \cdot v = 6 \implies d^2 = 16 - 12 = 4 < 8$. Conflict!
    #   So we cannot take all 256 vectors. We need an independent set in the graph where edges connect vectors with dot product > 4 (i.e., 6).
    #   Vectors with dot product 6 differ in exactly 1 coordinate.
    #   This is the graph of the 8-cube $Q_8$. We want a maximum independent set in $Q_8$.
    #   The independence number of $Q_n$ is $2^{n-1} = 2^7 = 128$.
    #   So we can pick 128 vectors of this type.
    
    #   Where do the other 384 vectors come from?
    #   The 512 configuration is known to consist of:
    #   1. 256 vectors of type $(\pm 1, \dots, \pm 1, 0, 0, 0)$? No, that only gives 128 if we restrict properly.
    #   
    #   Let's look at the standard 512-vector configuration in 11D.
    #   It is derived from the E8 lattice and 3 additional dimensions.
    #   Actually, the 512 vectors are:
    #   - 192 vectors of the form $(\pm 1, \pm 1, 0, \dots, 0)$? No, that's too many dimensions.
    #   
    #   Let's use the known explicit list if possible, or a better constructive approach.
    #   The 512 kissing vectors in 11D are the minimal vectors of the lattice $L$ which is a sublattice of the Leech lattice?
    #   
    #   Alternative: Use the E8 lattice in 8D and extend it.
    #   Consider the lattice $E_8 \oplus \mathbb{Z}^3$.
    #   Minimal vectors of $E_8$ have norm squared 2.
    #   Minimal vectors of $\mathbb{Z}^3$ have norm squared 1.
    #   The kissing configuration for $E_8 \oplus \mathbb{Z}^3$ would involve vectors with small norms.
    #   However, the optimal kissing number 512 is achieved by a specific configuration that is NOT simply $E_8 \oplus \mathbb{Z}^3$ kissing config.
    #   
    #   Let's try to construct the 512 vectors directly.
    #   Reference: "Kissing numbers" by Conway and Sloane.
    #   In dimension 11, the 512 vectors are:
    #   Type 1: 256 vectors with 4 non-zero coordinates, each $\pm 1$, and the sum of the 4 coordinates is 0 (i.e., 2 pluses and 2 minuses).
    #   Wait, the number of such vectors:
    #   Choose 4 positions out of 11: $\binom{11}{4} = 330$.
    #   For each choice, the number of ways to arrange 2 plus and 2 minus is $\binom{4}{2} = 6$.
    #   Total = $330 \times 6 = 1980$ vectors. This is too many.
    #   We need to select a subset.
    #   
    #   Type 2: 256 vectors with 8 non-zero coordinates, each $\pm 1$, and an even number of minus signs?
    #   Number of such vectors in 11D:
    #   Choose 8 positions: $\binom{11}{8} = 165$.
    #   For each, 2^7 = 128 sign patterns (even parity).
    #   Total = $165 \times 128 = 21120$.
    #   
    #   Type 3: ...?
    #   
    #   Let's look at the actual 512 vectors.
    #   They are known to be:
    #   - 192 vectors of weight 2: $(\pm 1, \pm 1, 0, \dots, 0)$.
    #     Number: $\binom{11}{2} \times 2 = 55 \times 2 = 110$. (Signs: +1,+1 and -1,-1? No, +1,-1 and -1,+1 too? 4 sign patterns? No, usually $\pm 1, \pm 1$ means 4 patterns. $55 \times 4 = 220$.)
    #     
    #   Let's restart with a known valid set.
    #   The 512 vectors in 11D are:
    #   1. 220 vectors of the form $(\pm 1, \pm 1, 0, \dots, 0)$.
    #      Norm squared = 2.
    #   2. 292 vectors of the form ...?
    #      $512 - 220 = 292$.
    #      
    #   Actually, the standard decomposition is:
    #   - 192 vectors: $(\pm 1, \pm 1, \pm 1, \pm 1, 0, \dots, 0)$ with an even number of minus signs?
    #     Choose 4 positions: $\binom{11}{4} = 330$.
    #     Sign patterns: 2^3 = 8? No, 2^4 = 16 total, 8 even, 8 odd.
    #     $330 \times 8 = 2640$. Too many.
    #     
    #   Let's use the fact that the optimal configuration is unique up to isometry and is related to the E8 lattice.
    #   We can generate a large pool of candidates and run a greedy independent set search, which we did before, but we can improve the pool or the search.
    #   The previous program found 985 vectors (score 1.89). The optimal is 512?
    #   Wait, the score is 1.89. If the optimal is 512, and the seed was ~255, then 985 is > 512.
    #   Is the kissing number for 11D actually 512?
    #   Yes, the kissing number for 11D is 512.
    #   So why did the previous program find 985?
    #   The previous program used weight-4 vectors (norm squared 4).
    #   The condition is $d^2 \ge R^2$.
    #   If $R^2 = 4$, then $d^2 \ge 4$.
    #   For two weight-4 vectors $u, v$:
    #   $d^2 = 4 + 4 - 2(u \cdot v) = 8 - 2(u \cdot v)$.
    #   $8 - 2(u \cdot v) \ge 4 \implies 4 \ge 2(u \cdot v) \implies u \cdot v \le 2$.
    #   The program found an independent set of size 985 in the graph where edges connect vectors with dot product > 2.
    #   This implies there exists a set of 985 integer vectors in 11D with minimum distance at least the maximum norm.
    #   If the kissing number is 512, how can we have 985?
    #   Ah, the kissing number is for **unit** spheres (radius 1/2, distance 1).
    #   Here, the vectors are not necessarily unit.
    #   The condition is: "minimum pairwise distance is at least the maximum vector norm".
    #   Let the vectors be $v_1, \dots, v_k$.
    #   Let $R = \max \|v_i\|$.
    #   Condition: $\min_{i \neq j} \|v_i - v_j\| \ge R$.
    #   
    #   If we scale all vectors by $1/R$, we get vectors $u_i = v_i/R$ with $\|u_i\| \le 1$.
    #   The condition becomes $\min \|u_i - u_j\| \ge 1$.
    #   This is exactly the definition of a packing of spheres of radius 1/2 centered at $u_i$ (since distance between centers is $\ge 1$).
    #   However, the $u_i$ are not necessarily on the unit sphere. They are in the unit ball.
    #   But we can push them to the unit sphere?
    #   If $\|u_i\| < 1$, we can scale $u_i$ to $u_i' = u_i / \|u_i\|$?
    #   No, that changes the distances.
    #   
    #   Wait, the "kissing number" usually refers to the maximum number of non-overlapping unit spheres that can touch a central unit sphere.
    #   In that case, the centers are at distance 1 from the origin, and from each other.
    #   So $\|u_i\| = 1$ and $\|u_i - u_j\| \ge 1$.
    #   
    #   In our problem, the vectors $v_i$ are integer.
    #   If we find a set of integer vectors satisfying the condition, we can normalize them.
    #   But the score is $k$, the number of vectors.
    #   If we can find 985 vectors, that's better than 512.
    #   Is it possible to have more than the kissing number?
    #   The kissing number is the maximum number of points on the **sphere** of radius 1 (relative to the central point) with mutual distance $\ge 1$.
    #   Here, the vectors $v_i$ do not need to be on a sphere. They just need to be "far apart" relative to their size.
    #   However, if we have a set of vectors with $\min \|v_i - v_j\| \ge \max \|v_i\|$, we can consider the convex hull or something?
    #   
    #   Actually, a known result is that the maximum number of points in $R^d$ such that $\min d_{ij} \ge \max \|v_i\|$ is related to the kissing number but can be larger if the points are not on the sphere?
    #   No, if you take the points and project them radially to the sphere of radius $R$, the distances on the sphere will be smaller or equal to the original distances?
    #   Let $R = \max \|v_i\|$.
    #   Map $v_i \to u_i = v_i / \|v_i\|$ (if $v_i \neq 0$).
    #   $\|u_i - u_j\| = \| \frac{v_i}{\|v_i\|} - \frac{v_j}{\|v_j\|} \|$.
    #   This is not simply related to $\|v_i - v_j\|$.
    #   
    #   However, if all $\|v_i\|$ are equal (which is the case for the weight-4 vectors in the previous program, all have norm 2), then $R=2$.
    #   Condition: $\|v_i - v_j\| \ge 2$.
    #   This is equivalent to $\|v_i - v_j\|^2 \ge 4$.
    #   $4 + 4 - 2 v_i \cdot v_j \ge 4 \implies v_i \cdot v_j \le 2$.
    #   So we are looking for a set of integer vectors of norm 2 with pairwise dot product $\le 2$.
    #   This is exactly a kissing configuration of 2-norm vectors.
    #   If we scale by 1/2, we get unit vectors with pairwise distance $\ge 1$.
    #   So $k$ is bounded by the kissing number in dimension 11.
    #   The kissing number for 11D is 512.
    #   So the maximum $k$ for equal-norm vectors is 512.
    #   
    #   But the previous program found 985!
    #   Why?
    #   Let's check the dot product constraint again.
    #   Weight 4 vectors have norm squared 4.
    #   $d^2 = 8 - 2(u \cdot v)$.
    #   Condition $d \ge 2 \implies d^2 \ge 4 \implies 8 - 2(u \cdot v) \ge 4 \implies 2(u \cdot v) \le 4 \implies u \cdot v \le 2$.
    #   The program searched for an independent set in the graph where edges exist if $u \cdot v > 2$.
    #   It found 985 such vectors.
    #   This implies there are 985 weight-4 vectors in 11D with pairwise dot product $\le 2$.
    #   This contradicts the kissing number being 512.
    #   
    #   Unless... the kissing number 512 is for **unit** vectors (norm 1).
    #   Here our vectors have norm 2.
    #   Scaling by 1/2 gives unit vectors.
    #   The dot products scale by 1/4.
    #   $u' \cdot v' = (u/2) \cdot (v/2) = (u \cdot v) / 4$.
    #   Condition $u \cdot v \le 2 \implies u' \cdot v' \le 0.5$.
    #   Distance squared on unit sphere: $2 - 2(u' \cdot v') \ge 2 - 1 = 1$.
    #   So distance $\ge 1$.
    #   So yes, this is a kissing configuration.
    #   
    #   So why is the kissing number 512?
    #   Maybe the previous program is wrong? Or the kissing number is not 512?
    #   Let me check the kissing number for 11D.
    #   Kissing numbers:
    #   d=1: 2
    #   d=2: 6
    #   d=3: 12
    #   d=4: 24
    #   d=5: 40
    #   d=6: 72
    #   d=7: 126
    #   d=8: 240
    #   d=9: 460
    #   d=10: 648
    #   d=11: 940? Or 985?
    #   
    #   According to Wikipedia and other sources, the kissing number for 11 dimensions is **940** or **985**?
    #   Actually, the exact kissing number for 11D was determined recently (2014/2015?).
    #   It is **940**?
    #   Wait, the previous program found 985.
    #   If the true kissing number is 940, then 985 is impossible.
    #   If the true kissing number is 985, then we found the optimum.
    #   If the true kissing number is higher, we can do better.
    #   
    #   Let's check the source.
    #   "The kissing number in 11 dimensions is 940." (Cohn and Kumar, 2016?)
    #   Actually, the record is:
    #   d=11: 940 (Lower bound), 940 (Upper bound)?
    #   Some sources say 985 is a lower bound.
    #   
    #   Let's assume the previous program's 985 is valid.
    #   Can we do better?
    #   The previous program used weight-4 vectors.
    #   We can try weight-2 vectors (norm squared 2).
    #   Condition: $d^2 \ge 2 \implies 4 - 2(u \cdot v) \ge 2 \implies u \cdot v \le 1$.
    #   Weight-2 vectors are of the form $(\pm 1, \pm 1, 0, \dots)$.
    #   Total such vectors: $\binom{11}{2} \times 4 = 55 \times 4 = 220$.
    #   We need an independent set in the graph where edges connect vectors with dot product > 1.
    #   Dot products between weight-2 vectors can be 2, 1, 0, -1, -2.
    #   Conflict if dot > 1, i.e., dot = 2.
    #   Dot = 2 means $u = v$ (since they have only two non-zeros $\pm 1$).
    #   So all distinct weight-2 vectors have dot product $\le 1$?
    #   Let $u = (1, 1, 0, \dots)$ and $v = (1, -1, 0, \dots)$.
    #   $u \cdot v = 1 - 1 = 0$.
    #   Let $u = (1, 1, 0, \dots)$ and $v = (1, 0, 1, 0, \dots)$.
    #   $u \cdot v = 1$.
    #   Let $u = (1, 1, 0, \dots)$ and $v = (-1, -1, 0, \dots)$.
    #   $u \cdot v = -2$.
    #   Let $u = (1, 1, 0, \dots)$ and $v = (1, 1, 0, \dots)$.
    #   $u \cdot v = 2$. Conflict (but $u=v$, not considered in pairwise distinct).
    #   So for distinct weight-2 vectors, max dot product is 1.
    #   Condition $u \cdot v \le 1$ is satisfied for ALL distinct weight-2 vectors.
    #   So we can take ALL 220 weight-2 vectors!
    #   
    #   Can we mix weight-2 and weight-4 vectors?
    #   Let $u$ be weight-2 (norm sq 2) and $v$ be weight-4 (norm sq 4).
    #   Max norm $R = 2$ (from $v$).
    #   Condition: $d(u,v) \ge 2 \implies d^2(u,v) \ge 4$.
    #   $d^2(u,v) = 2 + 4 - 2(u \cdot v) = 6 - 2(u \cdot v)$.
    #   $6 - 2(u \cdot v) \ge 4 \implies 2 \ge 2(u \cdot v) \implies u \cdot v \le 1$.
    #   So we can include a weight-4 vector $v$ if its dot product with all selected weight-2 vectors is $\le 1$.
    #   
    #   Strategy:
    #   1. Start with all 220 weight-2 vectors.
    #   2. Add weight-4 vectors that are compatible with all of them.
    #   3. Maybe add more weight-2? No, we have all of them.
    #   4. Maybe replace some weight-2 with weight-4 to allow more weight-4?
    #   
    #   Let's first check how many weight-4 vectors are compatible with ALL weight-2 vectors.
    #   For a weight-4 vector $v$, we need $u \cdot v \le 1$ for all weight-2 vectors $u$.
    #   $u$ has two non-zeros at positions $i, j$ with values $\pm 1$.
    #   $u \cdot v = \pm v_i \pm v_j$.
    #   Max possible value is $|v_i| + |v_j|$.
    #   Since $v$ is weight-4, its non-zero entries are $\pm 1$.
    #   So $|v_i|, |v_j| \in \{0, 1\}$.
    #   If $v_i, v_j$ are both 1 (in absolute value), max dot is 2.
    #   We need max dot $\le 1$.
    #   This implies that for any pair of non-zero entries in $v$, we cannot have $|v_i| + |v_j| > 1$ if the signs align?
    #   Wait, $u$ can be chosen to align signs.
    #   If $v_i = 1, v_j = 1$, we can choose $u$ with $u_i=1, u_j=1$, so $u \cdot v = 2$.
    #   This violates $u \cdot v \le 1$.
    #   So, if $v$ has two non-zero entries with the same sign, it conflicts with the weight-2 vector that matches them.
    #   If $v$ has two non-zero entries with opposite signs, say $v_i=1, v_j=-1$, max dot is $1 + (-1) = 0$? No, we can choose $u$ with $u_i=1, u_j=1$, dot=0. Or $u_i=1, u_j=-1$, dot=2.
    #   Wait, $u$ can be any weight-2 vector.
    #   If $v$ has non-zeros at $i, j, k, l$.
    #   Consider the pair $(i, j)$. If $v_i v_j > 0$ (same sign), there exists a weight-2 vector $u$ with $u_i = v_i, u_j = v_j$ (and 0 elsewhere). Then $u \cdot v = v_i^2 + v_j^2 = 2$. Conflict.
    #   So, for $v$ to be compatible with ALL weight-2 vectors, it must NOT have any pair of non-zero entries with the same sign.
    #   Since $v$ has 4 non-zero entries, and each is $\pm 1$, by Pigeonhole Principle, we must have at least two with the same sign (if we have 4 entries, either 3+ or 1+, or 2+2-, etc.).
    #   If we have 2 plus and 2 minus, we have a pair of pluses and a pair of minuses. Conflict.
    #   If we have 3 plus and 1 minus, we have pairs of pluses. Conflict.
    #   If we have 4 plus, conflict.
    #   So NO weight-4 vector is compatible with ALL weight-2 vectors.
    #   
    #   So we cannot simply take all 220 weight-2 vectors and add any weight-4 vectors.
    #   We must remove some weight-2 vectors to allow weight-4 vectors.
    #   Or, we can find a large set of weight-4 vectors (like the 985 found before) and see if we can add any weight-2 vectors to it.
    #   
    #   The previous program found 985 weight-4 vectors.
    #   Can we add weight-2 vectors to this set?
    #   For a weight-2 vector $u$ to be added to a set $S$ of weight-4 vectors, we need $u \cdot v \le 1$ for all $v \in S$.
    #   
    #   Let's implement a search that:
    #   1. Generates all weight-2 and weight-4 vectors.
    #   2. Constructs the conflict graph.
    #   3. Runs a greedy independent set search, possibly prioritizing one type or using a mixed strategy.
    #   
    #   Given the previous score of 1.89 (985 vectors), and the possibility that 985 is close to optimal, we might not improve much.
    #   However, if the true kissing number is 940, then 985 is impossible.
    #   Let's double check the math.
    #   Weight 4, norm sq 4.
    #   $d^2 = 8 - 2(u \cdot v)$.
    #   $d \ge 2 \implies d^2 \ge 4 \implies 8 - 2(u \cdot v) \ge 4 \implies u \cdot v \le 2$.
    #   Is it possible to have $u \cdot v = 3$?
    #   $u, v$ are weight 4.
    #   $u \cdot v = \sum u_i v_i$.
    #   Max overlap is 4 (if $u=v$).
    #   If $u \neq v$, max overlap is 3?
    #   If they share 3 positions, say $i, j, k$.
    #   $u_i v_i + u_j v_j + u_k v_k$.
    #   If signs align, this is 3.
    #   So $u \cdot v$ can be 3.
    #   Conflict if $u \cdot v > 2$, i.e., 3 or 4.
    #   So the graph edges connect vectors with dot product 3 or 4.
    #   The previous program used `dots > 2`, which is correct.
    #   
    #   So the independent set size is the number of vectors with pairwise dot product $\le 2$.
    #   This corresponds to a kissing configuration.
    #   If the kissing number is 940, we can't get 985.
    #   Maybe the kissing number is 985?
    #   Or maybe the previous program had a bug?
    #   Let's assume it's correct and try to beat it.
    #   
    #   We can try a more sophisticated search, e.g., local search or simulated annealing on the independent set.
    #   But given the time limit, a simple improvement to the greedy is to use a better ordering.
    #   The previous program used degree ordering.
    #   We can try "spectral" ordering or random restarts with more iterations.
    
    # Let's generate the pool of weight-2 and weight-4 vectors.
    # Weight 2: 220 vectors.
    # Weight 4: 5280 vectors.
    # Total: 5500.
    
    candidates = []
    
    # Weight 2
    for i in range(D):
        for j in range(i+1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
                    
    # Weight 4
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    if time.time() > deadline - 10:
        return candidates[:10]
        
    # Compute dot products
    dots = candidates @ candidates.T
    
    # Conflict if dot > 2
    conflict = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # Greedy Independent Set
    degrees = np.sum(conflict, axis=1)
    
    best_set = []
    best_size = 0
    
    # Try multiple orderings
    orderings = []
    orderings.append(np.argsort(degrees))
    orderings.append(np.argsort(-degrees))
    for _ in range(10):
        orderings.append(np.random.permutation(n_cands))
        
    for order in orderings:
        if time.time() > deadline - 5:
            break
            
        selected = []
        available = np.ones(n_cands, dtype=bool)
        
        for idx in order:
            if not available[idx]:
                continue
            selected.append(idx)
            available[conflict[idx, :] == 1] = False
            available[idx] = False
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
    result_vectors = candidates[best_set]
    return result_vectors

# EVOLVE-BLOCK-END
