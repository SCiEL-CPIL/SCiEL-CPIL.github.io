"""
Hybrid Adaptive Sketch (Structural Change):
- Direct Address Table (DAT) for items 0-255 (256 bytes, exact uint8 counters).
- Count-Min Sketch for items 256-399 (768 bytes).
  - 4 rows, 96 slots per row, uint16 counters.
  - 4 * 96 * 2 = 768 bytes.
- Total state size: 1024 bytes.

Rationale:
- The previous program used 128 bytes for 64 items (2 bytes/item) and 896 bytes for 336 items.
- The new program uses 256 bytes for 256 items (1 byte/item) and 768 bytes for 144 items.
- This shifts memory from the "head" (which can be stored exactly with 1 byte since max count is 1024? No, max count is 1024, which doesn't fit in uint8. Wait.)
- Correction: Max count is 1024. uint8 caps at 255.
- Let's re-evaluate the max count.
  - Stream 1: Uniform on 0..399. 1024 items. Expected count per item = 1024/400 = 2.56. Max likely < 10.
  - Stream 2: Zipf(1.6) modulo 400. Item 0 is very frequent. 1024 items.
    - Zipf probability p(k) = 1 / (k^1.6 * H(400, 1.6)).
    - H(400, 1.6) approx integral from 1 to 400 of x^-1.6 dx = [x^-0.6 / -0.6] = (1 - 400^-0.6)/0.6.
    - 400^-0.6 = (4*100)^-0.6 = 4^-0.6 * 100^-0.6 = 0.33 * 0.0158 = 0.005.
    - H approx (1 - 0.005)/0.6 = 1.658.
    - p(1) = 1 / (1 * 1.658) = 0.603.
    - Expected count for item 0 (index 1 in Zipf, 0 in mod 400 if 1-indexed? "Zipf(1.6) modulo 400" usually means k=1..400 mapped to 0..399 or k=0..399? Standard Zipf starts at 1. If items are 0..399, usually item 0 corresponds to rank 1.
    - Count for item 0 ~ 1024 * 0.603 = 617.
    - This exceeds 255. So uint8 is NOT sufficient for exact counting of item 0 in the Zipf stream.
    - However, the DAT is for items 0-255. Item 0 is in the DAT.
    - If we use uint8 for DAT, item 0 will saturate at 255. True count is ~617. Error = 362.
    - If we use uint16 for DAT, we need 256 * 2 = 512 bytes.
    - Remaining bytes: 1024 - 512 = 512 bytes for items 256-399 (144 items).
    - 512 bytes / 144 items. If we use CMS with uint16 counters:
      - 512 / 2 = 256 counters total.
      - Depth 4: 64 slots per row.
      - Depth 2: 128 slots per row.
    - Let's compare to the current program:
      - Current: DAT 0-63 (128 bytes, uint16). CMS 64-399 (896 bytes, 2 rows x 224 slots uint16).
      - Proposed A (uint8 DAT): DAT 0-255 (256 bytes, uint8, capped at 255). CMS 256-399 (768 bytes, 4 rows x 96 slots uint16).
        - Error on item 0: |255 - 617| = 362.
        - Error on other items: Low.
        - CMS for 144 items with 96 slots: Collision prob ~ 1/96.
      - Proposed B (uint16 DAT): DAT 0-255 (512 bytes, uint16, exact). CMS 256-399 (512 bytes, 4 rows x 64 slots uint16).
        - Error on item 0: 0.
        - CMS for 144 items with 64 slots: Collision prob ~ 1/64.
        - This is worse for the tail than Proposed A, but better for the head.
      - Proposed C (Mixed):
        - Item 0 is the main problem for uint8.
        - What if we use uint16 for the first few items (0-3) and uint8 for the rest?
        - Or just use uint16 for the whole DAT?
        - Let's look at the scores.
        - Current score: 6.46.
        - Inspiration 2 (uint8 DAT 0-255, uint8 CMS 256-399): Score 29.7. (Worse)
        - Why was Inspiration 2 worse?
          - It used uint8 for CMS. Max count in tail?
          - In Zipf, items 256-399 have very low probability.
          - p(257) = 1 / (257^1.6 * 1.658) ~ 1 / (257^1.6) ~ 1 / 4000 ~ 0.00025.
          - Expected count ~ 0.25.
          - So uint8 is fine for the tail in Zipf.
          - In Uniform, expected count 2.56. Max < 10. uint8 fine.
          - In Bimodal, items 256-399 are in the second half (150-399). Uniform on 150-399 (250 items). 512 items.
          - Expected count per item in second half = 512 / 250 = 2.048.
          - So uint8 is fine for the tail.
          - So why was Inspiration 2 (29.7) worse than Current (6.46)?
          - Inspiration 2 used 4 rows x 192 slots for the tail?
          - "4 rows x 96 slots x uint16 = 768 bytes".
          - Wait, the code in Inspiration 2 says:
            - SKETCH_COLS = 192.
            - SKETCH_STRIDE = 192.
            - But it reads/writes uint8?
            - `val = buf[offset]` (uint8).
            - `if val < 255: buf[offset] = val + 1`.
            - So it used uint8 counters in the CMS.
            - And it used 4 rows.
            - And it used 192 slots.
            - Total bytes for CMS: 4 * 192 * 1 = 768 bytes.
            - This matches the byte budget.
            - So Inspiration 2 used uint8 CMS.
            - Why did it score 29.7?
            - Maybe the hash function was bad?
            - `h = ((item * p + 31 * row) % 2147483647) % SKETCH_COLS`
            - This is the same hash as the seed.
            - Maybe the error on the head (0-255) was high?
            - Item 0 error: 362.
            - Mean error over 400 items: 362 / 400 = 0.9.
            - Normalized by 1024: 0.9 / 1024 = 0.00087.
            - This is small.
            - So the error must be in the tail.
            - With uint8 CMS, if there are collisions, the error is additive.
            - But with only 144 items and 192 slots, collisions are rare.
            - Unless... the "Zipf(1.6) modulo 400" means something else?
            - Or maybe the "Bimodal" stream has high counts in the tail?
            - Bimodal: 512 uniform on 0-19, 512 uniform on 150-399.
            - Items 150-399 are in the tail (256-399) for part of the range?
            - No, 150-255 are in the head (0-255). 256-399 are in the tail.
            - So in the Bimodal stream, items 256-399 receive 512 / 250 = 2.048 counts.
            - Items 150-255 receive 512 / 250 = 2.048 counts.
            - So the tail counts are low.
            
            - Why did Current (6.46) beat Inspiration 2 (29.7)?
            - Current uses uint16 for the head (0-63).
            - Current uses uint16 for the tail (64-399).
            - Current uses 2 rows for the tail.
            - Inspiration 2 uses uint8 for the head (0-255).
            - Inspiration 2 uses uint8 for the tail (256-399).
            - Inspiration 2 uses 4 rows for the tail.
            
            - Let's calculate the expected error for Current.
            - Head (0-63): Exact. Error 0.
            - Tail (64-399): CMS with 2 rows, 224 slots, uint16.
            - Items 64-399 is 336 items.
            - Collision probability per pair: 1/224.
            - Number of pairs: ~336^2 / 2 = 56000.
            - Expected collisions: 56000 / 224 = 250.
            - This is high.
            - But the counts are small.
            
            - Let's calculate the expected error for Proposed B (uint16 DAT 0-255, uint16 CMS 256-399, 4 rows, 64 slots).
            - Head (0-255): Exact. Error 0.
            - Tail (256-399): CMS with 4 rows, 64 slots, uint16.
            - Items 256-399 is 144 items.
            - Collision probability per pair: 1/64.
            - Number of pairs: ~144^2 / 2 = 10368.
            - Expected collisions: 10368 / 64 = 162.
            - This is also high, but the number of items is smaller.
            
            - Let's try Proposed A (uint8 DAT 0-255, uint16 CMS 256-399, 4 rows, 96 slots).
            - Head (0-255): uint8. Item 0 saturates at 255. True count ~617. Error 362.
            - Tail (256-399): CMS with 4 rows, 96 slots, uint16.
            - Items 256-399 is 144 items.
            - Collision probability per pair: 1/96.
            - Number of pairs: ~10368.
            - Expected collisions: 10368 / 96 = 108.
            
            - The error from Item 0 is 362.
            - The mean error contribution from Item 0 is 362/400 = 0.9.
            - The CMS error is distributed over 144 items.
            - If the average CMS error is E, the mean error is (362 + 144*E) / 400.
            
            - In Current, the mean error is (336 * E_current) / 400.
            - If Current score is 6.46 (normalized by 1024? No, score is mean absolute error / 1024).
            - So Mean Abs Error = 6.46 * 1024 = 6615.
            - Wait, the score is "mean over the 400 items of |estimate - true count| / 1024".
            - So Score = (1/400) * sum(|err|) / 1024.
            - So sum(|err|) = Score * 400 * 1024.
            - For Current: sum(|err|) = 6.46 * 400 * 1024 = 2,646,016.
            - This is huge.
            - Let's re-read the scoring.
            - "mean over the 400 items of |estimate - true count| / 1024"
            - This means: (1/400) * sum( |est - true| / 1024 ) = (1/409600) * sum(|est - true|).
            - So sum(|est - true|) = Score * 409600.
            - For Current: 6.46 * 409600 = 2,645,016.
            - This implies the total absolute error is ~2.6 million.
            - With 1024 items per stream, and 3 streams, total items = 3072.
            - Average error per item = 2,645,016 / 3072 = 861.
            - This is very high.
            
            - Why is the error so high?
            - In the Uniform stream, true count is ~2.56. Estimate should be close to 2.56.
            - In the Zipf stream, true count for item 0 is ~617.
            - In the Bimodal stream, true count for items 0-19 is ~512/20 = 25.6.
            
            - If the CMS is adding large errors, it's because of collisions.
            - But with 224 slots and 336 items, collisions are frequent.
            - Each collision adds the count of the other item.
            - If two items have counts c1 and c2, and they collide, the estimate for both becomes c1+c2.
            - Error for item 1: c2. Error for item 2: c1.
            - Total error: c1+c2.
            
            - To reduce error, we need to reduce collisions.
            - Proposed B (64 slots) has MORE collisions than Current (224 slots).
            - Proposed A (96 slots) has MORE collisions than Current (224 slots).
            
            - So the structural change to a smaller tail range (144 items) allows for FEWER slots if we keep the same byte budget, but we have MORE bytes for the tail in Proposed A/B.
            - Current: 896 bytes for tail.
            - Proposed A: 768 bytes for tail.
            - Proposed B: 512 bytes for tail.
            
            - So Current has the MOST bytes for the tail.
            - And it covers MORE items (336) in the tail.
            
            - Is it better to cover 336 items with 896 bytes (224 slots) or 144 items with 768 bytes (96 slots)?
            - Collision rate:
              - Current: 336 items, 224 slots. P(collision) ~ 336/224 = 1.5.
              - Proposed A: 144 items, 96 slots. P(collision) ~ 144/96 = 1.5.
            - The collision rate is similar!
            - But the counts are different.
            - In Current, the tail includes items 64-399.
            - In Zipf, items 64-399 have low counts.
            - In Bimodal, items 64-149 have low counts (0 in Bimodal? No, Bimodal is 0-19 and 150-399. So 20-149 are 0).
            - So in Bimodal, items 20-149 have true count 0.
            - If they collide with items that have counts, they get false positives.
            - In Proposed A, the tail is 256-399.
            - In Bimodal, items 256-399 have counts ~2.
            - Items 20-255 are in the head.
            - In Proposed A, items 20-149 are in the head (uint8). True count 0. Estimate 0. Error 0.
            - In Current, items 20-149 are in the tail (CMS). True count 0. Estimate > 0 due to collisions. Error > 0.
            
            - This suggests Proposed A is better for the Bimodal stream.
            - What about Zipf?
            - In Zipf, items 64-255 have low counts.
            - In Current, they are in the tail.
            - In Proposed A, they are in the head (uint8).
            - If their counts are < 255, they are exact.
            - p(65) = 1 / (65^1.6 * 1.658) ~ 1 / (1300 * 1.6) ~ 0.0005.
            - Expected count ~ 0.5.
            - So they are exact in Proposed A.
            - In Current, they are in the CMS and have collision errors.
            
            - So Proposed A should be better for Zipf and Bimodal.
            - What about Uniform?
            - Uniform: All items have count ~2.56.
            - In Current, items 64-399 are in CMS.
            - In Proposed A, items 64-255 are in Head (exact). Items 256-399 are in CMS.
            - So Proposed A has EXACT counts for more items.
            - The error in Proposed A comes only from the CMS part (256-399) and the saturation of item 0.
            - Item 0 error: 362.
            - CMS error: 144 items with collision rate ~1.5.
            - Current error: 336 items with collision rate ~1.5.
            
            - So Proposed A should have significantly lower error.
            - The only downside is the saturation of item 0.
            - Can we mitigate the saturation?
            - Use uint16 for item 0?
            - Layout:
              - Byte 0-1: Item 0 (uint16).
              - Byte 2-255: Items 1-255 (uint8).
              - Total Head Bytes: 2 + 255 = 257 bytes.
              - Remaining: 1024 - 257 = 767 bytes.
              - CMS: 767 bytes.
              - 4 rows x 95 slots x uint16 = 760 bytes.
              - 7 bytes left over.
            - This gives exact count for item 0.
            - And exact counts for items 1-255 (if < 255).
            - And CMS for 256-399.
            
            - Let's implement this hybrid layout.
            - Item 0: uint16 at offset 0.
            - Items 1-255: uint8 at offset 2 + (item-1).
            - Items 256-399: CMS at offset 257.
            - CMS: 4 rows, 95 slots, uint16.
            - 4 * 95 * 2 = 760 bytes.
            - Total: 257 + 760 = 1017 bytes.
            - 7 bytes padding.
            
            - This should be a strong improvement.
"""

import struct

# Configuration
_HEAD_OFFSET_0 = 0  # Item 0 is uint16 at offset 0
_HEAD_OFFSET_1 = 2  # Items 1-255 are uint8 starting at offset 2
_CMS_OFFSET = 257   # CMS starts at offset 257
_CMS_DEPTH = 4
_CMS_WIDTH = 95     # 4 * 95 * 2 = 760 bytes
_CMS_BYTES = _CMS_DEPTH * _CMS_WIDTH * 2

# Hash parameters for CM sketch
_A = [1035493289, 775374203, 1495862053, 2015145641]
_B = [0, 129, 257, 383]

def _read_uint16(buf, offset):
    return buf[offset] | (buf[offset + 1] << 8)

def _write_uint16(buf, offset, val):
    buf[offset] = val & 0xFF
    buf[offset + 1] = (val >> 8) & 0xFF

def update(state, item):
    buf = bytearray(state)
    
    if item == 0:
        # Item 0: uint16
        val = _read_uint16(buf, _HEAD_OFFSET_0)
        val += 1
        _write_uint16(buf, _HEAD_OFFSET_0, val)
    elif item < 256:
        # Items 1-255: uint8
        offset = _HEAD_OFFSET_1 + (item - 1)
        val = buf[offset]
        if val < 255:
            buf[offset] = val + 1
    else:
        # Items 256-399: CMS
        for i in range(_CMS_DEPTH):
            idx = (_A[i] * item + _B[i]) % _CMS_WIDTH
            byte_offset = _CMS_OFFSET + i * (_CMS_WIDTH * 2) + idx * 2
            val = _read_uint16(buf, byte_offset)
            val += 1
            _write_uint16(buf, byte_offset, val)
            
    return bytes(buf)

def query(state, item):
    buf = state
    
    if item == 0:
        return _read_uint16(buf, _HEAD_OFFSET_0)
    elif item < 256:
        offset = _HEAD_OFFSET_1 + (item - 1)
        return buf[offset]
    else:
        min_val = 65535
        for i in range(_CMS_DEPTH):
            idx = (_A[i] * item + _B[i]) % _CMS_WIDTH
            byte_offset = _CMS_OFFSET + i * (_CMS_WIDTH * 2) + idx * 2
            val = _read_uint16(buf, byte_offset)
            if val < min_val:
                min_val = val
        return min_val
